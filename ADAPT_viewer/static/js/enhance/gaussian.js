/**
 * Gaussian Blur implementation using separable kernels.
 * 
 * @param {Float32Array} data - The input 2D data array (flattened).
 * @param {number} width - Width of the data.
 * @param {number} height - Height of the data.
 * @param {number} sigma - Standard deviation of the Gaussian distribution.
 * @returns {Float32Array} - The blurred data.
 */
export function gaussianBlur(data, width, height, sigma) {
    if (sigma <= 0) return new Float32Array(data);

    const radius = Math.ceil(sigma * 3);
    const kernelSize = 2 * radius + 1;
    const kernel = new Float32Array(kernelSize);
    const twoSigmaSq = 2 * sigma * sigma;
    const sqrtTwoPiSigma = Math.sqrt(2 * Math.PI) * sigma;

    // Generate kernel
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
        const val = Math.exp(-(i * i) / twoSigmaSq) / sqrtTwoPiSigma;
        kernel[i + radius] = val;
        sum += val;
    }

    // Normalize kernel
    for (let i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }

    const temp = new Float32Array(data.length);
    const result = new Float32Array(data.length);

    // Horizontal pass
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let val = 0;
            let weightSum = 0;
            for (let k = -radius; k <= radius; k++) {
                const px = x + k;
                if (px >= 0 && px < width) {
                    const weight = kernel[k + radius];
                    val += data[y * width + px] * weight;
                    weightSum += weight;
                }
            }
            // Normalize for edge cases where kernel is truncated
            temp[y * width + x] = val / weightSum;
        }
    }

    // Vertical pass
    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
            let val = 0;
            let weightSum = 0;
            for (let k = -radius; k <= radius; k++) {
                const py = y + k;
                if (py >= 0 && py < height) {
                    const weight = kernel[k + radius];
                    val += temp[py * width + x] * weight;
                    weightSum += weight;
                }
            }
            result[y * width + x] = val / weightSum;
        }
    }

    return result;
}
