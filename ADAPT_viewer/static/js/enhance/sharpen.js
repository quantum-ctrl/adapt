import { gaussianBlur } from './gaussian.js';

/**
 * Sharpen implementation using Unsharp Mask.
 * Formula: result = original + strength * (original - blurred)
 * 
 * @param {Float32Array} data - The input 2D data array.
 * @param {number} width - Width of the data.
 * @param {number} height - Height of the data.
 * @param {number} strength - Strength of the sharpening (alpha).
 * @returns {Float32Array} - The sharpened data.
 */
export function sharpen(data, width, height, strength) {
    if (strength <= 0) return new Float32Array(data);

    // Use a fixed sigma multiplier for the unsharp mask blur
    // Typically, a small blur is used to define the "unsharp" version.
    // The user prompt suggested: blur = gaussianBlur(data, strength * 1.5)
    // However, strength is usually an amplitude factor, not a radius.
    // But the prompt explicitly said:
    // blur = gaussianBlur(data, strength * 1.5)
    // result = data + strength * (data - blur)
    // Wait, if strength is 0, sigma is 0. If strength is 2, sigma is 3.
    // This couples the radius of the unsharp mask to the strength of the sharpening.
    // This is a bit unusual (usually radius and amount are separate), but I will follow the prompt's formula.

    const sigma = Math.max(0.5, strength * 1.5);
    const blurred = gaussianBlur(data, width, height, sigma);
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
        result[i] = data[i] + strength * (data[i] - blurred[i]);
    }

    return result;
}
