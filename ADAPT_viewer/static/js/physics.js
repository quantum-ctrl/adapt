// Physics Constants
const CONSTANTS = {
    m_e: 9.10938356e-31,      // electron mass (kg)
    hbar: 1.054571817e-34,    // reduced Planck constant (J·s)
    eV_to_J: 1.602176634e-19, // electron volt to joules
    angstrom: 1e-10           // angstrom to meters
};

/**
 * Calculate kinetic energy from binding energy (relative to Fermi level).
 * Assumes energyValue is (E - E_F), so it's negative for occupied states.
 * E_k = (hv - phi) + energyValue
 * 
 * @param {number} energyValue - Energy relative to Fermi level (eV)
 * @param {number} hv - Photon energy (eV)
 * @param {number} workFunc - Work function (eV)
 * @returns {number} Kinetic energy (eV)
 */
export function calculateKineticEnergy(energyValue, hv, workFunc) {
    // If energyValue is E - E_F:
    // E_F_kinetic approx hv - phi
    // E_k = E_F_kinetic + (E - E_F) = hv - phi + energyValue
    return (hv - workFunc) + energyValue;
}

/**
 * Calculate momentum magnitude k (in inverse Angstroms).
 * k = sqrt(2 * m_e * E_k) / hbar
 * 
 * @param {number} kineticEnergy - Kinetic energy (eV)
 * @returns {number} k (1/Å)
 */
export function calculateK(kineticEnergy) {
    if (kineticEnergy < 0) return 0; // Should not happen for valid photoemission

    const Ek_J = kineticEnergy * CONSTANTS.eV_to_J;
    const k_m = Math.sqrt(2 * CONSTANTS.m_e * Ek_J) / CONSTANTS.hbar;

    // Convert to 1/Å
    return k_m * CONSTANTS.angstrom;
}

/**
 * Calculate parallel momentum component k_parallel.
 * k_par = k * sin(theta)
 * 
 * @param {number} k - Momentum magnitude (1/Å)
 * @param {number} angle - Emission angle (degrees)
 * @returns {number} k_parallel (1/Å)
 */
export function calculateKParallel(k, angle) {
    const theta_rad = angle * Math.PI / 180.0;
    return k * Math.sin(theta_rad);
}

/**
 * Calculate perpendicular momentum component k_z.
 * k_z = sqrt(2 * m_e * (E_k + V0)) / hbar * cos(theta)
 * 
 * @param {number} kineticEnergy - Kinetic energy (eV)
 * @param {number} innerPot - Inner potential V0 (eV)
 * @param {number} angle - Emission angle (degrees)
 * @returns {number} k_z (1/Å)
 */
export function calculateKz(kineticEnergy, innerPot, angle) {
    const E_inner = kineticEnergy + innerPot;
    if (E_inner < 0) return 0;

    const E_inner_J = E_inner * CONSTANTS.eV_to_J;
    const k_tot_m = Math.sqrt(2 * CONSTANTS.m_e * E_inner_J) / CONSTANTS.hbar;
    const k_tot_A = k_tot_m * CONSTANTS.angstrom;

    const theta_rad = angle * Math.PI / 180.0;
    return k_tot_A * Math.cos(theta_rad);
}

/**
 * Warp 2D image from (Angle, Energy) to (k_parallel, Energy).
 * 
 * @param {Float32Array} data - 2D data array (flattened, row-major)
 * @param {number} width - Number of angle points
 * @param {number} height - Number of energy points
 * @param {Array<number>} angleAxis - Array of angle values (degrees)
 * @param {Array<number>} energyAxis - Array of energy values (eV, relative to Fermi)
 * @param {number} hv - Photon energy (eV)
 * @param {number} workFunc - Work function (eV)
 * @returns {Object} { data: Float32Array, kAxis: Array<number>, width: number, height: number }
 */
export function warpToKSpace(data, width, height, angleAxis, energyAxis, hv, workFunc) {
    // 1. Determine k-range
    // Find min and max k across the entire dataset
    let minK = Infinity;
    let maxK = -Infinity;

    // Check corners is usually enough, but let's check all energies at min/max angle
    // Actually, k depends on E and Angle.
    // k ~ sqrt(E) * sin(theta).
    // Max k is at max E and max |theta|.
    // Min k is at max E and min theta (most negative).

    const angles = [angleAxis[0], angleAxis[width - 1]];
    const energies = [energyAxis[0], energyAxis[height - 1]];

    for (let E of energies) {
        const Ek = calculateKineticEnergy(E, hv, workFunc);
        const k_mag = calculateK(Ek);
        for (let ang of angles) {
            const k_par = calculateKParallel(k_mag, ang);
            if (k_par < minK) minK = k_par;
            if (k_par > maxK) maxK = k_par;
        }
    }

    // 2. Create uniform k-axis
    // Keep same number of points as angle axis? Or more?
    // Let's keep same width to preserve resolution roughly.
    const newWidth = width;
    const kAxis = new Float32Array(newWidth);
    const kStep = (maxK - minK) / (newWidth - 1);

    for (let i = 0; i < newWidth; i++) {
        kAxis[i] = minK + i * kStep;
    }

    // 3. Interpolate
    const newData = new Float32Array(newWidth * height);

    // For each row (Energy):
    for (let i = 0; i < height; i++) {
        const E = energyAxis[i];
        const Ek = calculateKineticEnergy(E, hv, workFunc);
        const k_mag = calculateK(Ek);

        // For this energy, k_par = k_mag * sin(theta)
        // We want to find value at k_target.
        // k_target = k_mag * sin(theta_target)
        // sin(theta_target) = k_target / k_mag
        // theta_target = asin(k_target / k_mag)

        // We iterate over the NEW k-grid
        for (let j = 0; j < newWidth; j++) {
            const k_target = kAxis[j];

            // Check if k_target is physically possible for this energy
            // |k_target| must be <= k_mag
            if (Math.abs(k_target) > k_mag) {
                newData[i * newWidth + j] = 0; // Out of bounds (evanescent?)
                continue;
            }

            const sin_theta = k_target / k_mag;
            const theta_target = Math.asin(sin_theta) * 180.0 / Math.PI;

            // Now interpolate from original angleAxis at theta_target
            // Find index in angleAxis
            // Assuming angleAxis is sorted (monotonic)
            // We can use binary search or simple linear search since it's small.
            // Or since we iterate k, theta is also monotonic.
            // Let's use a simple helper or assume uniform angle axis for speed?
            // Spec says "inconsistent axis ordering", but `load_data.py` creates linspace if delta != 0.
            // So angleAxis is likely uniform.

            // Map theta_target to index
            const angleStart = angleAxis[0];
            const angleEnd = angleAxis[width - 1];
            const angleRange = angleEnd - angleStart;

            // Normalized position (0 to 1)
            const t = (theta_target - angleStart) / angleRange;
            const idx = t * (width - 1);

            if (idx < 0 || idx > width - 1) {
                newData[i * newWidth + j] = 0;
                continue;
            }

            // Linear interpolation
            const idx0 = Math.floor(idx);
            const idx1 = Math.min(idx0 + 1, width - 1);
            const f = idx - idx0;

            const val0 = data[i * width + idx0];
            const val1 = data[i * width + idx1];

            newData[i * newWidth + j] = val0 * (1 - f) + val1 * f;
        }
    }

    return {
        data: newData,
        kAxis: kAxis,
        width: newWidth,
        height: height
    };
}

/**
 * Warp 3D data (Energy, AngleX, AngleY) to (Energy, kx, ky).
 * Assumes fixed photon energy.
 * 
 * @param {Float32Array} data - Flat 3D array [Energy, AngleX, AngleY]
 * @param {Array<number>} shape - [d0, d1, d2] -> [Energy, AngleX, AngleY]
 * @param {Object} axes - { dim0: energy, dim1: angleX, dim2: angleY }
 * @param {number} hv - Photon energy (eV)
 * @param {number} workFunc - Work function (eV)
 * @returns {Object} { data, shape, axes: { kx, ky, energy } }
 */
export function warp3D_KxKy(data, shape, axes, hv, workFunc) {
    const [nE, nAx, nAy] = shape;
    const energyAxis = axes.dim0;
    const angleXAxis = axes.dim1;
    const angleYAxis = axes.dim2;

    // 1. Determine global kx, ky range
    let minKx = Infinity, maxKx = -Infinity;
    let minKy = Infinity, maxKy = -Infinity;

    // Check corners of Energy and Angles
    const energies = [energyAxis[0], energyAxis[nE - 1]];
    const anglesX = [angleXAxis[0], angleXAxis[nAx - 1]];
    const anglesY = [angleYAxis[0], angleYAxis[nAy - 1]];

    for (let E of energies) {
        const Ek = calculateKineticEnergy(E, hv, workFunc);
        const k_tot = calculateK(Ek);

        for (let ax of anglesX) {
            const kx = calculateKParallel(k_tot, ax);
            if (kx < minKx) minKx = kx;
            if (kx > maxKx) maxKx = kx;
        }
        for (let ay of anglesY) {
            const ky = calculateKParallel(k_tot, ay);
            if (ky < minKy) minKy = ky;
            if (ky > maxKy) maxKy = ky;
        }
    }

    // 2. Create uniform k grids
    const nKx = nAx;
    const nKy = nAy;
    const kxAxis = new Float32Array(nKx);
    const kyAxis = new Float32Array(nKy);

    const kxStep = (maxKx - minKx) / (nKx - 1);
    const kyStep = (maxKy - minKy) / (nKy - 1);

    for (let i = 0; i < nKx; i++) kxAxis[i] = minKx + i * kxStep;
    for (let i = 0; i < nKy; i++) kyAxis[i] = minKy + i * kyStep;

    // 3. Warp
    const newData = new Float32Array(nE * nKx * nKy);

    for (let i = 0; i < nE; i++) {
        const E = energyAxis[i];
        const Ek = calculateKineticEnergy(E, hv, workFunc);
        const k_tot = calculateK(Ek);

        for (let j = 0; j < nKx; j++) {
            const kx = kxAxis[j];
            // Check physical bounds
            if (Math.abs(kx) > k_tot) continue; // 0

            const theta_x = Math.asin(kx / k_tot) * 180.0 / Math.PI;

            // Map theta_x to index in angleXAxis
            // Assuming linear axes for speed, else use binary search
            // Let's implement a quick linear map assuming sorted
            const axStart = angleXAxis[0];
            const axRange = angleXAxis[nAx - 1] - axStart;
            const u = (theta_x - axStart) / axRange;
            const idxX = u * (nAx - 1);

            if (idxX < 0 || idxX > nAx - 1) continue;

            const ix0 = Math.floor(idxX);
            const ix1 = Math.min(ix0 + 1, nAx - 1);
            const fx = idxX - ix0;

            for (let k = 0; k < nKy; k++) {
                const ky = kyAxis[k];
                // Check bounds (approximate for rectangular k-grid, strictly kx^2+ky^2 <= k_tot^2?)
                // Actually k_parallel = sqrt(kx^2 + ky^2).
                // But here we treat them as separable angles? 
                // Usually ARPES manipulators rotate theta and phi.
                // If theta_x and theta_y are independent rotations:
                // kx = k * sin(theta_x)
                // ky = k * sin(theta_y) ?? Depends on geometry.
                // Assuming simple geometry where kx and ky are derived from two angles.
                // Let's assume independent for now.

                if (Math.abs(ky) > k_tot) continue;

                const theta_y = Math.asin(ky / k_tot) * 180.0 / Math.PI;

                const ayStart = angleYAxis[0];
                const ayRange = angleYAxis[nAy - 1] - ayStart;
                const v = (theta_y - ayStart) / ayRange;
                const idxY = v * (nAy - 1);

                if (idxY < 0 || idxY > nAy - 1) continue;

                const iy0 = Math.floor(idxY);
                const iy1 = Math.min(iy0 + 1, nAy - 1);
                const fy = idxY - iy0;

                // Bilinear Interpolation
                // data is [E, Ax, Ay] -> i * (nAx*nAy) + ix * nAy + iy
                const base = i * nAx * nAy;

                const v00 = data[base + ix0 * nAy + iy0];
                const v01 = data[base + ix0 * nAy + iy1];
                const v10 = data[base + ix1 * nAy + iy0];
                const v11 = data[base + ix1 * nAy + iy1];

                const val = (v00 * (1 - fx) * (1 - fy)) +
                    (v01 * (1 - fx) * fy) +
                    (v10 * fx * (1 - fy)) +
                    (v11 * fx * fy);

                newData[i * nKx * nKy + j * nKy + k] = val;
            }
        }
    }

    return {
        data: newData,
        shape: [nE, nKx, nKy],
        axes: {
            dim0: energyAxis,
            dim1: kxAxis,
            dim2: kyAxis
        }
    };
}

/**
 * Warp 3D data (Energy, Angle, hv) to (Energy, kx, kz) or (Energy, kx, hv).
 * 
 * @param {Float32Array} data - Flat 3D array [Energy, Angle, hv]
 * @param {Array<number>} shape - [d0, d1, d2] -> [Energy, Angle, hv]
 * @param {Object} axes - { dim0: energy, dim1: angle, dim2: hv }
 * @param {number} workFunc - Work function (eV)
 * @param {number} innerPot - Inner potential (eV)
 * @param {boolean} keepHv - If true, map to (kx, hv), else (kx, kz)
 * @returns {Object} { data, shape, axes: { energy, kx, kz/hv } }
 */
export function warp3D_KxKz(data, shape, axes, workFunc, innerPot, keepHv) {
    const [nE, nA, nHv] = shape;
    const energyAxis = axes.dim0;
    const angleAxis = axes.dim1;
    const hvAxis = axes.dim2;

    // 1. Determine kx range (and kz range if converting)
    let minKx = Infinity, maxKx = -Infinity;
    let minKz = Infinity, maxKz = -Infinity;

    const energies = [energyAxis[0], energyAxis[nE - 1]];
    const angles = [angleAxis[0], angleAxis[nA - 1]];
    const hvs = [hvAxis[0], hvAxis[nHv - 1]];

    for (let E of energies) {
        for (let h of hvs) {
            const Ek = calculateKineticEnergy(E, h, workFunc);
            const k_tot = calculateK(Ek);

            for (let ang of angles) {
                const kx = calculateKParallel(k_tot, ang);
                if (kx < minKx) minKx = kx;
                if (kx > maxKx) maxKx = kx;

                if (!keepHv) {
                    const kz = calculateKz(Ek, innerPot, ang);
                    if (kz < minKz) minKz = kz;
                    if (kz > maxKz) maxKz = kz;
                }
            }
        }
    }

    const nKx = nA;
    const kxAxis = new Float32Array(nKx);
    const kxStep = (maxKx - minKx) / (nKx - 1);
    for (let i = 0; i < nKx; i++) kxAxis[i] = minKx + i * kxStep;

    let nDim2 = nHv;
    let dim2Axis = hvAxis; // Default if keepHv

    if (!keepHv) {
        nDim2 = nHv; // Keep same resolution
        dim2Axis = new Float32Array(nDim2);
        const kzStep = (maxKz - minKz) / (nDim2 - 1);
        for (let i = 0; i < nDim2; i++) dim2Axis[i] = minKz + i * kzStep;
    }

    const newData = new Float32Array(nE * nKx * nDim2);

    // 3. Warp
    for (let i = 0; i < nE; i++) {
        const E = energyAxis[i];

        for (let j = 0; j < nKx; j++) {
            const kx = kxAxis[j];

            for (let k = 0; k < nDim2; k++) {
                let targetHv, targetAngle;

                if (keepHv) {
                    // Target: (E, kx, hv)
                    targetHv = dim2Axis[k]; // hv is dim2
                    // Find Angle
                    const Ek = calculateKineticEnergy(E, targetHv, workFunc);
                    const k_tot = calculateK(Ek);

                    if (Math.abs(kx) > k_tot) continue;
                    targetAngle = Math.asin(kx / k_tot) * 180.0 / Math.PI;

                } else {
                    // Target: (E, kx, kz)
                    const kz = dim2Axis[k];

                    // Inverse Kinematics
                    // Ek_inner = (hbar^2 / 2m) * (kx^2 + kz^2)
                    // But we have constants.
                    // k_tot_inner^2 = kx^2 + kz^2 (in 1/A)
                    // E_inner = (hbar^2 * k_tot_inner_m^2) / 2m
                    // Let's use constants carefully.

                    const k_tot_sq = kx * kx + kz * kz; // 1/A^2
                    const k_tot_m = Math.sqrt(k_tot_sq) / CONSTANTS.angstrom;
                    const E_inner_J = (CONSTANTS.hbar * CONSTANTS.hbar * k_tot_m * k_tot_m) / (2 * CONSTANTS.m_e);
                    const E_inner = E_inner_J / CONSTANTS.eV_to_J;

                    const Ek = E_inner - innerPot;
                    if (Ek < 0) continue;

                    // hv = Ek + Phi - E
                    targetHv = Ek + workFunc - E;

                    // Angle
                    // kx = k_tot_out * sin(theta)
                    // k_tot_out = sqrt(2m Ek)/hbar
                    const k_tot_out_m = Math.sqrt(2 * CONSTANTS.m_e * Ek * CONSTANTS.eV_to_J) / CONSTANTS.hbar;
                    const k_tot_out = k_tot_out_m * CONSTANTS.angstrom;

                    if (Math.abs(kx) > k_tot_out) continue;
                    targetAngle = Math.asin(kx / k_tot_out) * 180.0 / Math.PI;
                }

                // Interpolate (Angle, hv)
                // Angle is dim1, hv is dim2

                // Map targetAngle to index
                const aStart = angleAxis[0];
                const aRange = angleAxis[nA - 1] - aStart;
                const u = (targetAngle - aStart) / aRange;
                const idxA = u * (nA - 1);

                if (idxA < 0 || idxA > nA - 1) continue;

                // Map targetHv to index
                const hStart = hvAxis[0];
                const hRange = hvAxis[nHv - 1] - hStart;
                const v = (targetHv - hStart) / hRange;
                const idxH = v * (nHv - 1);

                if (idxH < 0 || idxH > nHv - 1) continue;

                const iA0 = Math.floor(idxA);
                const iA1 = Math.min(iA0 + 1, nA - 1);
                const fA = idxA - iA0;

                const iH0 = Math.floor(idxH);
                const iH1 = Math.min(iH0 + 1, nHv - 1);
                const fH = idxH - iH0;

                // data is [E, A, hv]
                const base = i * nA * nHv;

                const v00 = data[base + iA0 * nHv + iH0];
                const v01 = data[base + iA0 * nHv + iH1];
                const v10 = data[base + iA1 * nHv + iH0];
                const v11 = data[base + iA1 * nHv + iH1];

                const val = (v00 * (1 - fA) * (1 - fH)) +
                    (v01 * (1 - fA) * fH) +
                    (v10 * fA * (1 - fH)) +
                    (v11 * fA * fH);

                newData[i * nKx * nDim2 + j * nDim2 + k] = val;
            }
        }
    }

    return {
        data: newData,
        shape: [nE, nKx, nDim2],
        axes: {
            dim0: energyAxis,
            dim1: kxAxis,
            dim2: dim2Axis
        }
    };
}

/**
 * Estimate Fermi level for 2D datasets (angle/k vs energy).
 * - data: Float32Array length = width * height, row-major with energy as rows (height)
 * - width: number of k (angle) points
 * - height: number of energy points
 * - energyAxis: Array of length height (eV)
 *
 * Returns: { ef, shift, status, data, energyAxis }
 *  - ef: detected EF value (eV)
 *  - shift: deltaE (ef - 0)
 *  - status: "success" | "fallback_used" | "failed"
 *  - data: shifted/resampled Float32Array (same shape)
 *  - energyAxis: updated energy axis (shifted)
 */
export function detectAndAlignEF2D(data, width, height, energyAxis, opts = {}) {
    // Options
    const smoothing = opts.smoothing || 1; // small smoothing window in points
    const minPeakSlope = opts.minPeakSlope || 1e-6;

    // 1) Compute EDC: sum over k for each energy
    const edc = new Float32Array(height);
    for (let i = 0; i < height; i++) {
        let s = 0;
        const rowStart = i * width;
        for (let j = 0; j < width; j++) s += data[rowStart + j];
        edc[i] = s;
    }

    // 2) Optional smoothing (simple moving average)
    const smoothEDC = new Float32Array(height);
    if (smoothing > 0 && smoothing < height) {
        const half = Math.floor(smoothing);
        for (let i = 0; i < height; i++) {
            let cnt = 0;
            let s = 0;
            for (let k = Math.max(0, i - half); k <= Math.min(height - 1, i + half); k++) {
                s += edc[k];
                cnt++;
            }
            smoothEDC[i] = s / Math.max(1, cnt);
        }
    } else {
        for (let i = 0; i < height; i++) smoothEDC[i] = edc[i];
    }

    // 3) Compute derivative along energy (dEDC/dE) using central differences
    const deriv = new Float32Array(height);
    for (let i = 1; i < height - 1; i++) {
        const de = energyAxis[i + 1] - energyAxis[i - 1];
        deriv[i] = de === 0 ? 0 : (smoothEDC[i + 1] - smoothEDC[i - 1]) / de;
    }
    deriv[0] = deriv[1];
    deriv[height - 1] = deriv[height - 2];

    // 4) Find maximum positive slope index
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < height; i++) {
        if (deriv[i] > maxVal) {
            maxVal = deriv[i];
            maxIdx = i;
        }
    }

    // If derivative peak is too small or at edge, fallback to simple max of EDC
    let ef = null;
    let status = "success";

    if (maxVal > minPeakSlope && maxIdx > 0 && maxIdx < height - 1) {
        // Interpolate peak position using quadratic around maxIdx for sub-bin precision
        const i = maxIdx;
        const y0 = deriv[i - 1], y1 = deriv[i], y2 = deriv[i + 1];
        const denom = (y0 - 2 * y1 + y2);
        let delta = 0;
        if (denom !== 0) delta = 0.5 * (y0 - y2) / denom;
        const efIndex = i + delta;
        // Map index to energy via linear interpolation on axis
        const eLo = energyAxis[Math.max(0, Math.floor(efIndex))];
        const eHi = energyAxis[Math.min(height - 1, Math.ceil(efIndex))];
        const t = efIndex - Math.floor(efIndex);
        ef = eLo * (1 - t) + eHi * t;
    } else {
        // Fallback: pick EDC maximum
        let maxEDC = -Infinity;
        let maxEDCIdx = 0;
        for (let i = 0; i < height; i++) {
            if (edc[i] > maxEDC) {
                maxEDC = edc[i];
                maxEDCIdx = i;
            }
        }
        if (maxEDCIdx >= 0 && maxEDCIdx < height) {
            ef = energyAxis[maxEDCIdx];
            status = "fallback_used";
        } else {
            return { ef: null, shift: 0, status: "failed" };
        }
    }

    // Compute shift ΔE = ef - 0
    const deltaE = ef;

    // 5) Resample data along energy: newData[iE, k] = interp(originalEnergy, originalColumn, energyAxis[iE] + deltaE)
    const newData = new Float32Array(width * height);
    for (let j = 0; j < width; j++) {
        // For each k-column, build column array
        // We'll do linear interpolation on original axis
        for (let i = 0; i < height; i++) {
            const targetE = energyAxis[i] + deltaE; // sample original at shifted position
            // Find bracketing indices
            // Quick clamp
            if (targetE <= energyAxis[0]) {
                newData[i * width + j] = data[0 * width + j];
                continue;
            }
            if (targetE >= energyAxis[height - 1]) {
                newData[i * width + j] = data[(height - 1) * width + j];
                continue;
            }
            // Binary search for index
            let lo = 0, hi = height - 1;
            while (hi - lo > 1) {
                const mid = (lo + hi) >> 1;
                if (energyAxis[mid] <= targetE) lo = mid;
                else hi = mid;
            }
            const e0 = energyAxis[lo], e1 = energyAxis[hi];
            const v0 = data[lo * width + j], v1 = data[hi * width + j];
            const w = (e1 === e0) ? 0 : (targetE - e0) / (e1 - e0);
            newData[i * width + j] = v0 * (1 - w) + v1 * w;
        }
    }

    // Shift energy axis values
    const newEnergyAxis = energyAxis.map(e => e - deltaE);

    return {
        ef: ef,
        shift: deltaE,
        status: status,
        data: newData,
        energyAxis: newEnergyAxis
    };
}

/**
 * Fit Fermi-Dirac edge to 2D data (sum over k -> EDC) and return EF.
 * Uses grid search over EF and kT with linear least-squares for A and B.
 *
 * Returns: { ef, status }
 */
export function fitFermiEdge2D(data, width, height, energyAxis, opts = {}) {
    const smoothing = opts.smoothing || 1;
    const window = opts.window || 0.6; // eV half-window around initial guess
    const efRange = opts.efRange || 0.6; // search +/- around initial

    // 1) Compute EDC
    const edc = new Float32Array(height);
    for (let i = 0; i < height; i++) {
        let s = 0;
        const row = i * width;
        for (let j = 0; j < width; j++) s += data[row + j];
        edc[i] = s;
    }

    // 2) Smooth (moving average)
    const smooth = new Float32Array(height);
    const half = Math.floor(smoothing);
    if (smoothing > 0 && smoothing < height) {
        for (let i = 0; i < height; i++) {
            let cnt = 0, ss = 0;
            for (let k = Math.max(0, i - half); k <= Math.min(height - 1, i + half); k++) {
                ss += edc[k]; cnt++;
            }
            smooth[i] = ss / Math.max(1, cnt);
        }
    } else {
        for (let i = 0; i < height; i++) smooth[i] = edc[i];
    }

    // 3) Initial EF guess via derivative peak
    const deriv = new Float32Array(height);
    for (let i = 1; i < height - 1; i++) {
        const de = energyAxis[i + 1] - energyAxis[i - 1];
        deriv[i] = de === 0 ? 0 : (smooth[i + 1] - smooth[i - 1]) / de;
    }
    deriv[0] = deriv[1]; deriv[height - 1] = deriv[height - 2];
    let maxIdx = 0, maxVal = -Infinity;
    for (let i = 0; i < height; i++) {
        if (deriv[i] > maxVal) { maxVal = deriv[i]; maxIdx = i; }
    }
    let initEF = energyAxis[maxIdx];

    // Search window indices
    const eMin = energyAxis[0], eMax = energyAxis[height - 1];
    const searchLow = Math.max(eMin, initEF - efRange);
    const searchHigh = Math.min(eMax, initEF + efRange);

    // Precompute arrays for windowed fit
    const energyArr = [];
    const intensityArr = [];
    for (let i = 0; i < height; i++) {
        const e = energyAxis[i];
        if (e >= searchLow && e <= searchHigh) {
            energyArr.push(e);
            intensityArr.push(smooth[i]);
        }
    }
    if (energyArr.length < 5) return { ef: initEF, status: 'fallback' };

    // Grid search parameters
    const efSteps = opts.efSteps || 61;
    const kTSteps = opts.kTSteps || 20;
    const efGrid = new Float32Array(efSteps);
    for (let i = 0; i < efSteps; i++) efGrid[i] = searchLow + (searchHigh - searchLow) * (i / (efSteps - 1));
    const kTMin = opts.kTMin || 0.001;
    const kTMax = opts.kTMax || 0.1;
    const kTGrid = new Float32Array(kTSteps);
    for (let i = 0; i < kTSteps; i++) kTGrid[i] = kTMin + (kTMax - kTMin) * (i / (kTSteps - 1));

    // Helper to compute residual for given EF,kT by solving for A,B (linear)
    const N = energyArr.length;
    let best = { rss: Infinity, ef: initEF, kT: 0.02, A: 0, B: 0 };

    // Precompute energy arrays as Float64 for numeric stability
    const E = energyArr;
    const Y = intensityArr;

    for (let ie = 0; ie < efGrid.length; ie++) {
        const efCandidate = efGrid[ie];
        for (let ik = 0; ik < kTGrid.length; ik++) {
            const kT = kTGrid[ik];
            // Build f vector
            let sum_f = 0, sum_1 = N, sum_ff = 0, sum_f1 = 0, sum_fy = 0, sum_1y = 0;
            for (let i = 0; i < N; i++) {
                const fi = 1.0 / (1.0 + Math.exp((E[i] - efCandidate) / kT));
                const yi = Y[i];
                sum_f += fi;
                sum_ff += fi * fi;
                sum_f1 += fi;
                sum_fy += fi * yi;
                sum_1y += yi;
            }
            // Normal equations for [A B] solving:
            // [sum_ff sum_f1][A] = [sum_fy]
            // [sum_f1  sum_11][B]   [sum_1y]  where sum_11 = N
            const det = sum_ff * N - sum_f1 * sum_f1;
            if (Math.abs(det) < 1e-12) continue;
            const A = (sum_fy * N - sum_f1 * sum_1y) / det;
            const B = (sum_ff * sum_1y - sum_f1 * sum_fy) / det;
            // compute rss
            let rss = 0;
            for (let i = 0; i < N; i++) {
                const fi = 1.0 / (1.0 + Math.exp((E[i] - efCandidate) / kT));
                const pred = A * fi + B;
                const d = Y[i] - pred;
                rss += d * d;
            }
            if (rss < best.rss) {
                best = { rss, ef: efCandidate, kT, A, B };
            }
        }
    }

    // Optional refine around best ef with smaller step
    const refineEFSteps = 31;
    const refineKSteps = 10;
    const efLow = Math.max(eMin, best.ef - (searchHigh - searchLow) / efSteps);
    const efHigh = Math.min(eMax, best.ef + (searchHigh - searchLow) / efSteps);
    for (let ie = 0; ie < refineEFSteps; ie++) {
        const efCandidate = efLow + (efHigh - efLow) * (ie / (refineEFSteps - 1));
        for (let ik = 0; ik < refineKSteps; ik++) {
            const kT = Math.max(kTMin, Math.min(kTMax, best.kT + (ik - refineKSteps / 2) * ((kTMax - kTMin) / (kTSteps * 2))));
            let sum_f = 0, sum_1 = N, sum_ff = 0, sum_f1 = 0, sum_fy = 0, sum_1y = 0;
            for (let i = 0; i < N; i++) {
                const fi = 1.0 / (1.0 + Math.exp((E[i] - efCandidate) / kT));
                const yi = Y[i];
                sum_f += fi;
                sum_ff += fi * fi;
                sum_f1 += fi;
                sum_fy += fi * yi;
                sum_1y += yi;
            }
            const det = sum_ff * N - sum_f1 * sum_f1;
            if (Math.abs(det) < 1e-12) continue;
            const A = (sum_fy * N - sum_f1 * sum_1y) / det;
            const B = (sum_ff * sum_1y - sum_f1 * sum_fy) / det;
            let rss = 0;
            for (let i = 0; i < N; i++) {
                const fi = 1.0 / (1.0 + Math.exp((E[i] - efCandidate) / kT));
                const pred = A * fi + B;
                const d = Y[i] - pred;
                rss += d * d;
            }
            if (rss < best.rss) {
                best = { rss, ef: efCandidate, kT, A, B };
            }
        }
    }

    if (!best || !isFinite(best.ef)) return { ef: initEF, status: 'failed' };
    return { ef: best.ef, status: 'success', fit: best };
}

/**
 * Complementary Error Function (approximate)
 * erfc(x) = 1 - erf(x)
 * Uses Abramowitz and Stegun approximation (max error 1.5e-7)
 */
function erfc(x) {
    // Save the sign of x
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);

    // Constants
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    // A&S formula 7.1.26
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    // erf(x) = sign * y
    // erfc(x) = 1 - erf(x)
    return 1.0 - sign * y;
}

/**
 * Enhanced fitFermiEdge2D with Gaussian Broadening support.
 * Model: I(E) = A * 0.5 * erfc( (E - E_F) / (sqrt(2) * sigma_tot) ) + B
 * Where sigma_tot = sqrt(sigma^2 + (pi*k_B*T/sqrt(3))^2)
 */
export function fitFermiEdgeGaussian(data, width, height, energyAxis, opts = {}) {
    const smoothing = opts.smoothing || 1;
    const window = opts.window || 0.1; // eV half-window for fitting range
    // NOTE: user might pass a smaller window (e.g. 0.1) for tight fit around edge
    const initialEF = opts.initialEF; // Must provide a starting guess
    if (initialEF === undefined || initialEF === null) {
        return { success: false, error: "Initial EF guess required" };
    }

    // Fixed T = 20K (approx 1.7 meV) or user provided
    const T = opts.temperature || 20.0;
    const kB = 8.617e-5;
    const kT = kB * T;

    // 1) Compute EDC and Normalized EDC in one pass
    const edc = new Float32Array(height);

    // Compute raw EDC
    for (let i = 0; i < height; i++) {
        let s = 0;
        const row = i * width;
        for (let j = 0; j < width; j++) s += data[row + j];
        edc[i] = s;
    }

    // Smooth EDC if needed
    const smooth = new Float32Array(height);
    let minVal = Infinity, maxVal = -Infinity;

    if (smoothing > 0) {
        const half = Math.floor(smoothing);
        for (let i = 0; i < height; i++) {
            let s = 0, cnt = 0;
            const start = Math.max(0, i - half);
            const end = Math.min(height - 1, i + half);
            for (let k = start; k <= end; k++) {
                s += edc[k];
                cnt++;
            }
            smooth[i] = s / cnt;
            if (smooth[i] < minVal) minVal = smooth[i];
            if (smooth[i] > maxVal) maxVal = smooth[i];
        }
    } else {
        for (let i = 0; i < height; i++) {
            smooth[i] = edc[i];
            if (smooth[i] < minVal) minVal = smooth[i];
            if (smooth[i] > maxVal) maxVal = smooth[i];
        }
    }

    // Normalize
    const normRange = (maxVal - minVal) > 1e-9 ? (maxVal - minVal) : 1;
    const normEDC = new Float32Array(height);
    for (let i = 0; i < height; i++) {
        normEDC[i] = (smooth[i] - minVal) / normRange;
    }

    // 2) Select Data within Fitting Window
    const eMin = initialEF - window;
    const eMax = initialEF + window;
    const X = [];
    const Y = [];

    for (let i = 0; i < height; i++) {
        const e = energyAxis[i];
        if (e >= eMin && e <= eMax) {
            X.push(e);
            Y.push(normEDC[i]);
        }
    }

    if (X.length < 5) return { success: false, error: "Not enough points in window" };

    // 3) Grid Search optimization
    const efSteps = 40;
    const sigmaSteps = 20;

    // Search ranges
    const searchEFMin = initialEF - window * 0.5;
    const searchEFMax = initialEF + window * 0.5;
    const sigmaMin = 0.001; // 1 meV
    const sigmaMax = 0.1;   // 100 meV

    let best = { rss: Infinity, ef: initialEF, sigma: 0.02, A: 1, B: 0 };
    const N = X.length;

    // Pre-calculate thermal width component squared
    const thermalWidthSq = Math.pow(Math.PI * kT / Math.sqrt(3), 2);

    for (let ie = 0; ie < efSteps; ie++) {
        const efVal = searchEFMin + (ie / (efSteps - 1)) * (searchEFMax - searchEFMin);

        for (let is = 0; is < sigmaSteps; is++) {
            const sigmaVal = sigmaMin + (is / (sigmaSteps - 1)) * (sigmaMax - sigmaMin);

            // Total width
            const totalWidth = Math.sqrt(sigmaVal * sigmaVal + thermalWidthSq);
            const sqrt2TotalWidth = Math.sqrt(2) * totalWidth;

            // Build model vectors for linear least squares
            let sum_f = 0, sum_ff = 0, sum_y = 0, sum_fy = 0;

            for (let k = 0; k < N; k++) {
                const arg = (X[k] - efVal) / sqrt2TotalWidth;
                const f = 0.5 * erfc(arg);
                const yVal = Y[k];

                sum_f += f;
                sum_ff += f * f;
                sum_y += yVal;
                sum_fy += f * yVal;
            }

            const det = sum_ff * N - sum_f * sum_f;
            if (Math.abs(det) < 1e-12) continue; // Singular

            const A = (sum_fy * N - sum_f * sum_y) / det;
            const B = (sum_ff * sum_y - sum_f * sum_fy) / det;

            // Calc RSS
            let rss = 0;
            for (let k = 0; k < N; k++) {
                const arg = (X[k] - efVal) / sqrt2TotalWidth;
                const f = 0.5 * erfc(arg);
                const pred = A * f + B;
                const resid = Y[k] - pred;
                rss += resid * resid;
            }

            if (rss < best.rss) {
                best = { rss, ef: efVal, sigma: sigmaVal, A, B };
            }
        }
    }

    // Refinement Step (Zoom in around best)
    const refineSteps = 20;
    const efRangeRefine = (searchEFMax - searchEFMin) / efSteps * 4;
    const sigmaRangeRefine = (sigmaMax - sigmaMin) / sigmaSteps * 4;

    const rEFMin = Math.max(eMin, best.ef - efRangeRefine);
    const rEFMax = Math.min(eMax, best.ef + efRangeRefine);
    const rSigmaMin = Math.max(sigmaMin, best.sigma - sigmaRangeRefine);
    const rSigmaMax = Math.min(sigmaMax, best.sigma + sigmaRangeRefine);

    for (let ie = 0; ie < refineSteps; ie++) {
        const efVal = rEFMin + (ie / (refineSteps - 1)) * (rEFMax - rEFMin);
        for (let is = 0; is < refineSteps; is++) {
            const sigmaVal = rSigmaMin + (is / (refineSteps - 1)) * (rSigmaMax - rSigmaMin);

            const totalWidth = Math.sqrt(sigmaVal * sigmaVal + thermalWidthSq);
            const sqrt2TotalWidth = Math.sqrt(2) * totalWidth;

            let sum_f = 0, sum_ff = 0, sum_y = 0, sum_fy = 0;
            for (let k = 0; k < N; k++) {
                const arg = (X[k] - efVal) / sqrt2TotalWidth;
                const f = 0.5 * erfc(arg);
                const yVal = Y[k];
                sum_f += f;
                sum_ff += f * f;
                sum_y += yVal;
                sum_fy += f * yVal;
            }

            const det = sum_ff * N - sum_f * sum_f;
            if (Math.abs(det) < 1e-12) continue;

            const A = (sum_fy * N - sum_f * sum_y) / det;
            const B = (sum_ff * sum_y - sum_f * sum_fy) / det;

            let rss = 0;
            for (let k = 0; k < N; k++) {
                const arg = (X[k] - efVal) / sqrt2TotalWidth;
                const f = 0.5 * erfc(arg);
                const pred = A * f + B;
                const resid = Y[k] - pred;
                rss += resid * resid;
            }
            if (rss < best.rss) {
                best = { rss, ef: efVal, sigma: sigmaVal, A, B };
            }
        }
    }

    return {
        success: true,
        ef: best.ef,
        sigma: best.sigma,
        width: Math.sqrt(best.sigma * best.sigma + thermalWidthSq), // Total Gaussian width
        rss: best.rss
    };
}
