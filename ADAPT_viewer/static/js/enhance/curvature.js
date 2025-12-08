/**
 * Curvature enhancement (2nd derivative method) for ARPES data.
 * 
 * @param {Float32Array} data - The input 2D data array (flattened, row-major).
 * @param {number} width - Width of the data.
 * @param {number} height - Height of the data.
 * @param {number} strength - Strength multiplier for the curvature.
 * @param {string} direction - 'x', 'y', or 'both'.
 * @returns {Float32Array} - The curvature-enhanced data.
 */
export function curvature2D(data, width, height, strength, direction = 'y') {
    if (strength <= 0) return new Float32Array(data);

    // Initialize result with original data to ensure all pixels have valid values
    const result = new Float32Array(data);

    // Compute second derivative
    // d²I/dx² for horizontal, d²I/dy² for vertical (energy direction)

    if (direction === 'x' || direction === 'both') {
        // Horizontal second derivative (along angle axis)
        for (let y = 0; y < height; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                const left = data[y * width + (x - 1)];
                const center = data[idx];
                const right = data[y * width + (x + 1)];
                // Second derivative: f''(x) ≈ f(x-1) - 2f(x) + f(x+1)
                const d2 = left - 2 * center + right;
                result[idx] = direction === 'both' ? -d2 * strength : center - d2 * strength;
            }
        }
        // Edge pixels (x=0 and x=width-1) keep original values from initialization
    }

    if (direction === 'y' || direction === 'both') {
        // Use result from x-pass if 'both', otherwise use original data
        const input = direction === 'both' ? new Float32Array(result) : data;

        // Vertical second derivative (along energy axis)
        for (let y = 1; y < height - 1; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;
                const top = input[(y - 1) * width + x];
                const center = input[idx];
                const bottom = input[(y + 1) * width + x];
                // Second derivative: f''(y) ≈ f(y-1) - 2f(y) + f(y+1)
                const d2 = top - 2 * center + bottom;
                result[idx] = center - d2 * strength;
            }
        }
        // Edge pixels (y=0 and y=height-1) keep original values from initialization
    }

    return result;
}

