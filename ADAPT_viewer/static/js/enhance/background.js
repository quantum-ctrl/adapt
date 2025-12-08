import { gaussianBlur } from './gaussian.js';

/**
 * Background Removal using Rolling-mean approximation (via Gaussian Blur).
 * Formula: result = data - gaussianBlur(data, radius)
 * 
 * @param {Float32Array} data - The input 2D data array.
 * @param {number} width - Width of the data.
 * @param {number} height - Height of the data.
 * @param {number} radius - The radius (sigma) for the background estimation.
 * @returns {Float32Array} - The data with background removed.
 */
export function removeBackground(data, width, height, radius) {
    if (radius <= 0) return new Float32Array(data);

    const background = gaussianBlur(data, width, height, radius);
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
        result[i] = data[i] - background[i];
    }

    return result;
}
