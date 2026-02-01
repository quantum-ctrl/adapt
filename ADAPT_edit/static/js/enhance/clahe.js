/**
 * CLAHE (Contrast Limited Adaptive Histogram Equalization) implementation.
 * 
 * @param {Float32Array} data - The input 2D data array.
 * @param {number} width - Width of the data.
 * @param {number} height - Height of the data.
 * @param {number} clipLimit - Contrast limit (normalized 0-1, typically small e.g. 0.01).
 * @returns {Float32Array} - The enhanced data.
 */
export function clahe2D(data, width, height, clipLimit) {
    if (clipLimit <= 0) return new Float32Array(data);

    // Configuration
    const numTilesX = 8;
    const numTilesY = 8;
    const numBins = 256;

    // 1. Find global min/max to normalize data to 0-1 for histogram building
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < data.length; i++) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
    }
    const range = maxVal - minVal;
    if (range === 0) return new Float32Array(data);

    // Helper to map value to bin index
    const getBin = (val) => {
        let bin = Math.floor(((val - minVal) / range) * numBins);
        if (bin < 0) bin = 0;
        if (bin >= numBins) bin = numBins - 1;
        return bin;
    };

    // 2. Divide image into tiles and calculate histograms
    const tileWidth = Math.ceil(width / numTilesX);
    const tileHeight = Math.ceil(height / numTilesY);

    // Histograms: [tileY][tileX][bin]
    const histograms = [];
    for (let ty = 0; ty < numTilesY; ty++) {
        histograms[ty] = [];
        for (let tx = 0; tx < numTilesX; tx++) {
            histograms[ty][tx] = new Uint32Array(numBins);
        }
    }

    for (let y = 0; y < height; y++) {
        const ty = Math.floor(y / tileHeight);
        for (let x = 0; x < width; x++) {
            const tx = Math.floor(x / tileWidth);
            const bin = getBin(data[y * width + x]);
            histograms[ty][tx][bin]++;
        }
    }

    // 3. Clip histograms and create CDFs
    // Actual clip limit in pixel count
    // Average pixels per bin = (tileWidth * tileHeight) / numBins
    // clipLimit input is usually a factor relative to something or a normalized value.
    // Standard CLAHE clip limit is often a slope limit in the CDF, or max count in histogram.
    // Let's assume clipLimit is normalized such that 1.0 is huge. 
    // OpenCV uses typically 2.0-4.0 as a value, but here we have 0-0.05 from prompt.
    // If 0.05 is the limit, maybe it means fraction of total pixels in the tile?
    // Let's interpret clipLimit as fraction of tile area.
    const clipLimitCount = Math.max(1, Math.floor(clipLimit * tileWidth * tileHeight));

    const cdfs = []; // [tileY][tileX][bin] -> normalized CDF value 0-1

    for (let ty = 0; ty < numTilesY; ty++) {
        cdfs[ty] = [];
        for (let tx = 0; tx < numTilesX; tx++) {
            const hist = histograms[ty][tx];

            // Clip
            let clippedCount = 0;
            for (let b = 0; b < numBins; b++) {
                if (hist[b] > clipLimitCount) {
                    clippedCount += hist[b] - clipLimitCount;
                    hist[b] = clipLimitCount;
                }
            }

            // Redistribute
            const redistBatch = Math.floor(clippedCount / numBins);
            const redistRemainder = clippedCount % numBins;

            for (let b = 0; b < numBins; b++) {
                hist[b] += redistBatch;
            }
            // Distribute remainder evenly (or just first few bins)
            for (let b = 0; b < redistRemainder; b++) {
                hist[b]++;
            }

            // Calculate CDF
            const cdf = new Float32Array(numBins);
            let sum = 0;
            const totalPixels = tileWidth * tileHeight; // Approximation (last tiles might be smaller but we normalize)
            // Actually better to use actual pixel count for this tile if we want precision, 
            // but tiles are mostly equal. Let's use the sum of the histogram which is accurate.
            let histSum = 0;
            for (let b = 0; b < numBins; b++) histSum += hist[b];

            for (let b = 0; b < numBins; b++) {
                sum += hist[b];
                cdf[b] = sum / histSum;
            }
            cdfs[ty][tx] = cdf;
        }
    }

    // 4. Interpolate
    const result = new Float32Array(data.length);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const val = data[y * width + x];
            const bin = getBin(val);

            // Find the four tiles surrounding this pixel
            // Center of a tile is at (tx * tileWidth + tileWidth/2, ty * tileHeight + tileHeight/2)

            // We want to interpolate based on tile centers.
            // Map x, y to "tile coordinates" where integer values are tile centers.
            // Tile center 0 is at tileWidth/2. Tile center 1 is at tileWidth + tileWidth/2.
            // coord = (x - tileWidth/2) / tileWidth

            const tx = (x - tileWidth / 2) / tileWidth;
            const ty = (y - tileHeight / 2) / tileHeight;

            const tx1 = Math.floor(tx);
            const tx2 = tx1 + 1;
            const ty1 = Math.floor(ty);
            const ty2 = ty1 + 1;

            // Weights
            const wx2 = tx - tx1;
            const wx1 = 1.0 - wx2;
            const wy2 = ty - ty1;
            const wy1 = 1.0 - wy2;

            // Helper to get CDF value safely
            const getCdfVal = (tX, tY, b) => {
                // Clamp tile indices
                if (tX < 0) tX = 0;
                if (tX >= numTilesX) tX = numTilesX - 1;
                if (tY < 0) tY = 0;
                if (tY >= numTilesY) tY = numTilesY - 1;
                return cdfs[tY][tX][b];
            };

            // Bilinear interpolation
            const val11 = getCdfVal(tx1, ty1, bin);
            const val12 = getCdfVal(tx1, ty2, bin);
            const val21 = getCdfVal(tx2, ty1, bin);
            const val22 = getCdfVal(tx2, ty2, bin);

            const interpolatedCdf =
                val11 * wx1 * wy1 +
                val21 * wx2 * wy1 +
                val12 * wx1 * wy2 +
                val22 * wx2 * wy2;

            // Map back to original range
            result[y * width + x] = minVal + interpolatedCdf * range;
        }
    }

    return result;
}
