import { CropBox } from './crop_box.js';
import { gaussianBlur } from './enhance/gaussian.js';
import { sharpen } from './enhance/sharpen.js';
import { removeBackground } from './enhance/background.js';
import { clahe2D } from './enhance/clahe.js';
import { curvature2D } from './enhance/curvature.js';


// Class constants
const MARGIN = { top: 10, right: 15, bottom: 45, left: 70 };
const PROFILE_PADDING = { left: 10, right: 10, top: 8, bottom: 28 };
const DEBUG = false; // Set to true to enable console.log statements

export class Visualizer {
    constructor(container) {
        this.container = container;
        this.data = null;
        this.metadata = null;
        this.axes = null;
        this.shape = null;
        this.ndim = 0;

        // State
        this.colormap = 'viridis';
        this.contrastMin = 0; // Percentile 0-100
        this.contrastMax = 100;
        this.inverted = false; // Colormap inversion flag
        this.sliceIndices = [0, 0, 0]; // Current slice indices [energy, angle, scan]
        this.axisLabels = { x: '', y: '', z: '' }; // Custom labels

        // Crop mode state
        this.mode = 'normal'; // 'normal' | 'crop'
        this.activeCropView = null; // null | 'xy' | 'xz' | 'yz'
        this.cropBox = null;
        this.originalDataBackup = null;
        this.originalMetaBackup = null;

        // Cache for colormaps and temp canvas
        this.colormapCache = {};
        this.currentLut = this.getColormap('viridis');
        this.tempCanvas = null; // Reusable temp canvas for rendering

        // Resize observer
        this.resizeObserver = new ResizeObserver(() => this.onResize());
        this.resizeObserver.observe(this.container);

        // Calibration Mode
        this.calibrationMode = 'none'; // 'none' | 'angle' | 'fermi'
        this.calibrationLinePos = 0; // Data coordinate for X or primary axis
        this.calibrationLinePosY = 0; // Data coordinate for Y axis (for 3D angle calibration)
        this.isDraggingCalibration = false;
        this.calibrationDragAxis = null; // 'x' | 'y' when dragging calibration lines
        this.calibrationPositions = {}; // keyed by axis name: { kx, ky, energy }
        // Persistent EF marker (shows detected EF even when not in calibration mode)
        this.efMarker = null; // energy value in data coordinates (eV) or null

        // Enhancement State
        this.enhancement = {
            enabled: false,
            smoothing: 0,
            sharpen: 0,
            background: 0,
            clahe: false,
            claheClip: 0.01,
            curvature: false,
            curvatureStrength: 1.0
        };
    }

    setEnhancement(opts) {
        if (DEBUG) console.log("Visualizer.setEnhancement:", opts);
        this.enhancement = { ...this.enhancement, ...opts };
        this.draw();
    }

    /**
     * Set a persistent EF marker line (energy coordinate).
     * Pass null to clear.
     */
    setEFMarker(value) {
        this.efMarker = (value === null || value === undefined) ? null : value;
        this.draw();
    }

    applyEnhancement(data, width, height) {
        if (!this.enhancement.enabled || !data || !(data instanceof Float32Array)) {
            return data;
        }

        if (DEBUG) console.log("Applying enhancement...", this.enhancement);
        let result = data; // Float32Array

        // 1. Smoothing
        if (this.enhancement.smoothing > 0) {
            result = gaussianBlur(result, width, height, this.enhancement.smoothing);
        }

        // 2. Sharpen
        if (this.enhancement.sharpen > 0) {
            result = sharpen(result, width, height, this.enhancement.sharpen);
        }

        // 3. Background Removal
        if (this.enhancement.background > 0) {
            result = removeBackground(result, width, height, this.enhancement.background);
        }

        // 4. CLAHE
        if (this.enhancement.clahe) {
            result = clahe2D(result, width, height, this.enhancement.claheClip);
        }

        // 5. Curvature (2nd derivative enhancement)
        if (this.enhancement.curvature) {
            result = curvature2D(result, width, height, this.enhancement.curvatureStrength, 'y');
        }

        return result;
    }

    /**
     * Set cursor position (slice indices) from physical coordinates.
     * @param {Object} coords - { x: theta/kx, y: energy/ef, z: scan/ky }
     */
    setCursor(coords) {
        if (!this.data || !this.axes) return;

        const findClosestIndex = (arr, val) => {
            if (!arr || arr.length === 0) return 0;
            let closestIdx = 0;
            let minDiff = Infinity;
            for (let i = 0; i < arr.length; i++) {
                const diff = Math.abs(arr[i] - val);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIdx = i;
                }
            }
            return closestIdx;
        };

        const newIndices = [...this.sliceIndices];
        let changed = false;

        // 2D: x=Angle(kx), y=Energy
        // 3D: x=Angle, y=Energy, z=Scan
        // Note: sliceIndices for 2D is [energy_idx, angle_idx, 0]
        // Note: sliceIndices for 3D is [energy_idx, angle_idx, scan_idx]

        if (coords.y !== undefined && this.axes.energy) {
            const idx = findClosestIndex(this.axes.energy, coords.y);
            if (idx !== newIndices[0]) {
                newIndices[0] = idx;
                changed = true;
            }
        }

        if (coords.x !== undefined && this.axes.kx) {
            const idx = findClosestIndex(this.axes.kx, coords.x);
            if (idx !== newIndices[1]) {
                newIndices[1] = idx;
                changed = true;
            }
        }

        if (this.ndim === 3 && coords.z !== undefined && this.axes.ky) {
            const idx = findClosestIndex(this.axes.ky, coords.z);
            if (idx !== newIndices[2]) {
                newIndices[2] = idx;
                changed = true;
            }
        }

        if (changed) {
            this.sliceIndices = newIndices;
            this.draw();
            // We do NOT dispatch 'cursor-update' here to avoid circular loop with UI inputs
            // unless we strictly distinguish source. For now, let's assume UI update -> setCursor
            // shouldn't trigger UI update again.
        }
    }

    setData(data, metadata) {
        this.data = data;
        this.metadata = metadata;
        this.axes = metadata.axes;
        this.shape = metadata.data_info.shape;
        this.ndim = metadata.data_info.ndim;

        // Initialize slice indices to middle
        if (this.ndim === 3) {
            this.sliceIndices = this.shape.map(d => Math.floor(d / 2));
        } else if (this.ndim === 2) {
            // For 2D: [energy_idx, angle_idx, 0] - always use 3 elements for consistency
            this.sliceIndices = [Math.floor(this.shape[0] / 2), Math.floor(this.shape[1] / 2), 0];
        }

        // Initialize calibration positions per physical axis so 3D per-axis calibration works.
        this.calibrationPositions = {};
        if (this.axes) {
            if (this.axes.kx && this.axes.kx.length > 0) {
                const kxMin = this.axes.kx[0];
                const kxMax = this.axes.kx[this.axes.kx.length - 1];
                this.calibrationPositions.kx = (kxMin + kxMax) / 2;
            }
            if (this.axes.ky && this.axes.ky.length > 0) {
                const kyMin = this.axes.ky[0];
                const kyMax = this.axes.ky[this.axes.ky.length - 1];
                this.calibrationPositions.ky = (kyMin + kyMax) / 2;
            }
            if (this.axes.energy && this.axes.energy.length > 0) {
                const eMin = this.axes.energy[0];
                const eMax = this.axes.energy[this.axes.energy.length - 1];
                this.calibrationPositions.energy = (eMin + eMax) / 2;
            }
        }

        this.renderLayout();
        this.draw();
    }

    renderLayout() {
        this.container.innerHTML = '';

        if (this.ndim === 2) {
            // 2D layout: Image left (60%), EDC/MDC right (40% stacked)
            this.container.innerHTML = `
                <div class="layout-2d">
                    <div class="image-panel" id="view-2d">
                        <canvas id="canvas-main"></canvas>
                        <div class="overlay-text">2D View</div>
                    </div>
                    <div class="profile-panel">
                        <div class="profile-cell" id="view-edc">
                            <canvas id="canvas-edc"></canvas>
                            <div class="profile-label">EDC</div>
                        </div>
                        <div class="profile-cell" id="view-mdc">
                            <canvas id="canvas-mdc"></canvas>
                            <div class="profile-label">MDC</div>
                        </div>
                    </div>
                </div>
            `;
            this.canvases = [this.container.querySelector('#canvas-main')];
            this.edcCanvas = this.container.querySelector('#canvas-edc');
            this.mdcCanvas = this.container.querySelector('#canvas-mdc');

            // Add event listeners to profile canvases for cursor control
            this.setupProfileInteraction(this.edcCanvas, 'edc');
            this.setupProfileInteraction(this.mdcCanvas, 'mdc');
        } else if (this.ndim === 3) {
            // 3D Grid view with profiles in lower-right
            this.container.innerHTML = `
                <div class="grid-3d">
                    <div class="grid-cell" id="view-xy">
                        <canvas data-plane="xy"></canvas>
                        <div class="overlay-text">XY (Z-Cut)</div>
                    </div>
                    <div class="grid-cell" id="view-xz">
                        <canvas data-plane="xz"></canvas>
                        <div class="overlay-text">XZ (Y-Cut)</div>
                    </div>
                    <div class="grid-cell" id="view-yz">
                        <canvas data-plane="yz"></canvas>
                        <div class="overlay-text">YZ (X-Cut)</div>
                    </div>
                    <div class="grid-cell profile-container" id="view-profiles">
                        <div class="profile-stack">
                            <div class="profile-row" id="profile-edc">
                                <canvas id="canvas-edc-3d"></canvas>
                                <span class="profile-label">EDC</span>
                            </div>
                            <div class="profile-row" id="profile-mdc">
                                <canvas id="canvas-mdc-3d"></canvas>
                                <span class="profile-label">MDC</span>
                            </div>
                            <div class="profile-row" id="profile-scan">
                                <canvas id="canvas-scan"></canvas>
                                <span class="profile-label">Scan</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            this.canvases = Array.from(this.container.querySelectorAll('canvas[data-plane]'));
            this.edcCanvas = this.container.querySelector('#canvas-edc-3d');
            this.mdcCanvas = this.container.querySelector('#canvas-mdc-3d');
            this.scanCanvas = this.container.querySelector('#canvas-scan');

            // Add event listeners to profile canvases for cursor control
            this.setupProfileInteraction(this.edcCanvas, 'edc');
            this.setupProfileInteraction(this.mdcCanvas, 'mdc');
            this.setupProfileInteraction(this.scanCanvas, 'scan');
        }

        // Attach event listeners to canvases
        this.canvases.forEach(canvas => {
            canvas.addEventListener('mousemove', (e) => this.handleInteraction(e, canvas));
            canvas.addEventListener('mousedown', (e) => this.handleInteraction(e, canvas));
            canvas.addEventListener('mouseup', (e) => this.handleInteraction(e, canvas));
            canvas.addEventListener('click', (e) => this.handleViewClick(e, canvas));
        });
    }

    draw() {
        if (!this.data) return;

        if (this.ndim === 2) {
            // 2D View: Energy (Y) vs Angle (X)
            // axes.energy is Y, axes.kx is X
            // sliceIndices[1] = angle index (X), sliceIndices[0] = energy index (Y)
            const crosshair2D = [this.sliceIndices[1], this.sliceIndices[0]];
            this.draw2D(
                this.canvases[0],
                this.data,
                this.shape[1], this.shape[0],
                this.axes.kx, this.axes.energy,
                this.axisLabels.x || "Angle", this.axisLabels.y || "Energy (eV)",
                crosshair2D
            );

            // Draw EDC and MDC profiles
            this.drawProfiles2D();
        } else if (this.ndim === 3) {
            const [nz, ny, nx] = this.shape;
            // shape[0]=Energy(Y), shape[1]=Angle(X), shape[2]=Scan(Z)

            const idxY = this.sliceIndices[0]; // Energy
            const idxX = this.sliceIndices[1]; // Angle
            const idxZ = this.sliceIndices[2]; // Scan

            // 1. XY View (Z-cut): Energy vs Angle
            const sliceXY = this.extractSlice(2, idxZ);
            this.draw2D(
                this.canvases[0],
                sliceXY,
                this.shape[1], this.shape[0],
                this.axes.kx, this.axes.energy,
                this.axisLabels.x || "Angle", this.axisLabels.y || "Energy (eV)",
                [idxX, idxY]
            );

            // 2. XZ View (Y-cut): Energy vs Scan - NOW TOP RIGHT
            // X-axis: Scan (dim2), Y-axis: Energy (dim0)
            const sliceXZ = this.extractSlice(1, idxX);
            this.draw2D(
                this.canvases[1],
                sliceXZ,
                this.shape[2], this.shape[0],
                this.axes.ky, this.axes.energy,
                this.axisLabels.z || "Scan", this.axisLabels.y || "Energy (eV)",
                [idxZ, idxY]
            );

            // 3. YZ View (X-cut): Angle vs Scan - NOW BOTTOM LEFT
            // X-axis: Scan (dim2), Y-axis: Angle (dim1)
            const sliceYZ = this.extractSlice(0, idxY);
            this.draw2D(
                this.canvases[2],
                sliceYZ,
                this.shape[2], this.shape[1],
                this.axes.ky, this.axes.kx,
                this.axisLabels.z || "Scan", this.axisLabels.x || "Angle",
                [idxZ, idxX]
            );

            // Draw profiles for 3D
            this.drawProfiles3D(sliceXY, idxX, idxY, idxZ);
        }
    }

    draw2D(canvas, data, dataWidth, dataHeight, xAxis, yAxis, xLabel, yLabel, crosshair = []) {
        // Get container size
        const rect = canvas.parentElement.getBoundingClientRect();
        const displayWidth = rect.width;
        const displayHeight = rect.height;

        // Set canvas resolution to match display size (times DPR for sharpness)
        const dpr = window.devicePixelRatio || 1;
        canvas.width = displayWidth * dpr;
        canvas.height = displayHeight * dpr;

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr); // Scale context so we draw in CSS pixels
        ctx.clearRect(0, 0, displayWidth, displayHeight);

        // Calculate Plot Area
        const plotWidth = displayWidth - MARGIN.left - MARGIN.right;
        const plotHeight = displayHeight - MARGIN.top - MARGIN.bottom;

        if (plotWidth <= 0 || plotHeight <= 0) return;

        // 1. Draw Heatmap
        // Reuse temp canvas for performance
        if (!this.tempCanvas) {
            this.tempCanvas = document.createElement('canvas');
        }
        const tempCanvas = this.tempCanvas;
        tempCanvas.width = dataWidth;
        tempCanvas.height = dataHeight;
        const tempCtx = tempCanvas.getContext('2d');
        const imgData = tempCtx.createImageData(dataWidth, dataHeight);
        const buf = imgData.data;
        const lut = this.currentLut;

        const globalMin = this.metadata.data_info.min;
        const globalMax = this.metadata.data_info.max;
        const range = globalMax - globalMin;
        const cMin = globalMin + (this.contrastMin / 100) * range;
        const cMax = globalMin + (this.contrastMax / 100) * range;
        const cRange = cMax - cMin || 1;

        // Apply Enhancement
        // Note: We process the data before mapping to colors.
        // This is expensive, so ideally we should cache if data/params haven't changed.
        // For now, we do it every frame as requested (client-side pipeline).
        const enhancedData = this.applyEnhancement(data, dataWidth, dataHeight);

        // Fill buffer (Native Data Resolution)
        // Data is row-major. data[0] is (0,0).
        // We want to draw it such that:
        // - If we draw normally, (0,0) is top-left.
        // - Physics usually wants (0,0) at bottom-left.
        // - So we flip Y when drawing to the main canvas, OR we fill the buffer flipped.
        // Let's fill buffer flipped so (0,0) is at bottom of the image.

        for (let i = 0; i < dataHeight; i++) {
            // i is row index (y-coordinate in data).
            // We want data-y=0 at the bottom of the image.
            // Image Y=0 is top.
            // So data-y=0 goes to image-y = height-1.
            const imgY = dataHeight - 1 - i;

            for (let j = 0; j < dataWidth; j++) {
                const val = enhancedData[i * dataWidth + j];

                let norm = (val - cMin) / cRange;
                if (norm < 0) norm = 0;
                if (norm > 1) norm = 1;

                const lutIdx = Math.floor(norm * 255) * 4;
                const bufIdx = (imgY * dataWidth + j) * 4;

                buf[bufIdx] = lut[lutIdx];
                buf[bufIdx + 1] = lut[lutIdx + 1];
                buf[bufIdx + 2] = lut[lutIdx + 2];
                buf[bufIdx + 3] = 255;
            }
        }

        tempCtx.putImageData(imgData, 0, 0);

        // Draw scaled image to main canvas
        // Disable smoothing for pixelated look
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tempCanvas,
            0, 0, dataWidth, dataHeight, // Source
            MARGIN.left, MARGIN.top, plotWidth, plotHeight // Destination
        );

        // 2. Draw Axes
        ctx.strokeStyle = '#888';
        ctx.fillStyle = '#ccc';
        ctx.font = '12px sans-serif';
        ctx.lineWidth = 1;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        // X Axis
        ctx.beginPath();
        ctx.moveTo(MARGIN.left, MARGIN.top + plotHeight);
        ctx.lineTo(MARGIN.left + plotWidth, MARGIN.top + plotHeight);
        ctx.stroke();

        // X Ticks
        if (xAxis && xAxis.length > 1) {
            const xMin = xAxis[0];
            const xMax = xAxis[xAxis.length - 1];
            const ticks = this.getNiceTicks(xMin, xMax, Math.floor(plotWidth / 60));

            ticks.forEach(val => {
                const t = (val - xMin) / (xMax - xMin);
                if (t >= 0 && t <= 1) {
                    const x = MARGIN.left + t * plotWidth;

                    ctx.beginPath();
                    ctx.moveTo(x, MARGIN.top + plotHeight);
                    ctx.lineTo(x, MARGIN.top + plotHeight + 5);
                    ctx.stroke();

                    let label;
                    if (Math.abs(val) < 1e-10) {
                        label = "0";
                    } else {
                        label = Math.abs(val) < 0.01 ? val.toExponential(1) : parseFloat(val.toFixed(2)).toString();
                    }
                    ctx.fillText(label, x, MARGIN.top + plotHeight + 8);
                }
            });

            // X Label
            ctx.fillStyle = '#e0e0e0';
            ctx.fillText(xLabel || "X", MARGIN.left + plotWidth / 2, MARGIN.top + plotHeight + 25);
        }

        // Y Axis
        ctx.beginPath();
        ctx.moveTo(MARGIN.left, MARGIN.top);
        ctx.lineTo(MARGIN.left, MARGIN.top + plotHeight);
        ctx.stroke();

        // Y Ticks
        if (yAxis && yAxis.length > 1) {
            const yMin = yAxis[0];
            const yMax = yAxis[yAxis.length - 1];
            const ticks = this.getNiceTicks(yMin, yMax, Math.floor(plotHeight / 40));

            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#ccc';

            ticks.forEach(val => {
                const t = (val - yMin) / (yMax - yMin);
                if (t >= 0 && t <= 1) {
                    // Y axis is inverted in plot (bottom is min, top is max)
                    const y = MARGIN.top + plotHeight - t * plotHeight;

                    ctx.beginPath();
                    ctx.moveTo(MARGIN.left, y);
                    ctx.lineTo(MARGIN.left - 5, y);
                    ctx.stroke();

                    let label;
                    if (Math.abs(val) < 1e-10) {
                        label = "0";
                    } else {
                        label = Math.abs(val) < 0.01 ? val.toExponential(1) : parseFloat(val.toFixed(2)).toString();
                    }
                    ctx.fillText(label, MARGIN.left - 8, y);
                }
            });

            // Y Label
            ctx.save();
            ctx.translate(15, MARGIN.top + plotHeight / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.fillStyle = '#e0e0e0';
            ctx.fillText(yLabel || "Y", 0, 0);
            ctx.restore();
        }

        // Persistent EF Marker (draw even when not in calibration mode)
        if (this.efMarker !== null && this.axes && this.axes.energy && this.axes.energy.length > 1) {
            ctx.save();
            ctx.strokeStyle = '#00e5ff'; // Cyan-ish marker
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);

            // Determine whether energy is on X or Y axis by label hints
            const lx = (xLabel || '').toLowerCase();
            const ly = (yLabel || '').toLowerCase();
            const xIsEnergy = lx.includes('energy') || lx.includes('ev');
            const yIsEnergy = ly.includes('energy') || ly.includes('ev');

            const drawVerticalAt = (val, axisArray, labelText) => {
                const xMin = axisArray[0];
                const xMax = axisArray[axisArray.length - 1];
                const t = (val - xMin) / (xMax - xMin);
                if (t >= 0 && t <= 1) {
                    const x = MARGIN.left + t * plotWidth;
                    ctx.beginPath();
                    ctx.moveTo(x, MARGIN.top);
                    ctx.lineTo(x, MARGIN.top + plotHeight);
                    ctx.stroke();
                    ctx.fillStyle = '#00e5ff';
                    ctx.textAlign = 'center';
                    ctx.fillText(labelText, x, MARGIN.top - 10);
                }
            };

            const drawHorizontalAt = (val, axisArray, labelText) => {
                const yMin = axisArray[0];
                const yMax = axisArray[axisArray.length - 1];
                const t = (val - yMin) / (yMax - yMin);
                if (t >= 0 && t <= 1) {
                    const y = MARGIN.top + plotHeight - t * plotHeight;
                    ctx.beginPath();
                    ctx.moveTo(MARGIN.left, y);
                    ctx.lineTo(MARGIN.left + plotWidth, y);
                    ctx.stroke();
                    ctx.fillStyle = '#00e5ff';
                    ctx.textAlign = 'right';
                    ctx.fillText(labelText, MARGIN.left - 10, y);
                }
            };

            if (xIsEnergy && xAxis && xAxis.length > 1) {
                drawVerticalAt(this.efMarker, xAxis, "EF");
            } else if (yIsEnergy && yAxis && yAxis.length > 1) {
                drawHorizontalAt(this.efMarker, yAxis, "EF");
            } else {
                // Default: assume energy is Y axis
                if (yAxis && yAxis.length > 1) drawHorizontalAt(this.efMarker, yAxis, "EF");
            }
            ctx.restore();
        }

        // 3. Draw Crosshair
        if (crosshair.length === 2) {
            const [cx, cy] = crosshair;
            // cx, cy are data indices.
            const tX = cx / (dataWidth - 1);
            const tY = cy / (dataHeight - 1);

            const canvasX = MARGIN.left + tX * plotWidth;
            const canvasY = MARGIN.top + plotHeight - tY * plotHeight;

            ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            // Vertical
            ctx.moveTo(canvasX, MARGIN.top);
            ctx.lineTo(canvasX, MARGIN.top + plotHeight);
            // Horizontal
            ctx.moveTo(MARGIN.left, canvasY);
            ctx.lineTo(MARGIN.left + plotWidth, canvasY);
            ctx.stroke();
        }

        // 5. Draw Calibration Line
        if (this.calibrationMode !== 'none') {
            ctx.save();
            ctx.strokeStyle = '#ffeb3b'; // Yellow
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 3]);

            // Determine axis types by labels (provided by App via setAxisLabels)
            const lx = (xLabel || '').toLowerCase();
            const ly = (yLabel || '').toLowerCase();
            const xIsAngle = lx.includes('angle') || lx.includes('deg');
            const yIsAngle = ly.includes('angle') || ly.includes('deg');
            const xIsEnergy = lx.includes('energy') || lx.includes('ev');
            const yIsEnergy = ly.includes('energy') || ly.includes('ev');

            // Helper to draw vertical line at data value along xAxis
            const drawVerticalAt = (val, axisArray, labelText) => {
                const xMin = axisArray[0];
                const xMax = axisArray[axisArray.length - 1];
                const t = (val - xMin) / (xMax - xMin);
                if (t >= 0 && t <= 1) {
                    const x = MARGIN.left + t * plotWidth;
                    ctx.beginPath();
                    ctx.moveTo(x, MARGIN.top);
                    ctx.lineTo(x, MARGIN.top + plotHeight);
                    ctx.stroke();
                    ctx.fillStyle = '#ffeb3b';
                    ctx.textAlign = 'center';
                    ctx.fillText(labelText, x, MARGIN.top - 10);
                }
            };

            // Helper to draw horizontal line at data value along yAxis
            const drawHorizontalAt = (val, axisArray, labelText) => {
                const yMin = axisArray[0];
                const yMax = axisArray[axisArray.length - 1];
                const t = (val - yMin) / (yMax - yMin);
                if (t >= 0 && t <= 1) {
                    const y = MARGIN.top + plotHeight - t * plotHeight;
                    ctx.beginPath();
                    ctx.moveTo(MARGIN.left, y);
                    ctx.lineTo(MARGIN.left + plotWidth, y);
                    ctx.stroke();
                    ctx.fillStyle = '#ffeb3b';
                    ctx.textAlign = 'right';
                    ctx.fillText(labelText, MARGIN.left - 10, y);
                }
            };

            // Helper to map axis array to calibration key
            const getAxisKey = (axisArr) => {
                if (axisArr === this.axes.kx) return 'kx';
                if (axisArr === this.axes.ky) return 'ky';
                if (axisArr === this.axes.energy) return 'energy';
                return null;
            };

            const getCalForAxis = (axisArr, fallbackVal) => {
                const k = getAxisKey(axisArr);
                if (k && this.calibrationPositions[k] !== undefined) return this.calibrationPositions[k];
                return fallbackVal;
            };

            if (this.calibrationMode === 'angle') {
                // Draw vertical line if this view's X axis is angle
                if (xIsAngle && xAxis && xAxis.length > 1) {
                    const val = getCalForAxis(xAxis, this.calibrationLinePos);
                    drawVerticalAt(val, xAxis, "Angle Zero (X)");
                }

                // Draw horizontal line if this view's Y axis is angle
                if (yIsAngle && yAxis && yAxis.length > 1) {
                    const val = getCalForAxis(yAxis, this.calibrationLinePosY);
                    drawHorizontalAt(val, yAxis, "Angle Zero (Y)");
                }
            } else if (this.calibrationMode === 'fermi') {
                // If energy is on X axis, draw vertical; if on Y axis, draw horizontal.
                if (xIsEnergy && xAxis && xAxis.length > 1) {
                    const val = getCalForAxis(xAxis, this.calibrationLinePos);
                    drawVerticalAt(val, xAxis, "EF");
                }
                if (yIsEnergy && yAxis && yAxis.length > 1) {
                    const val = getCalForAxis(yAxis, this.calibrationLinePos);
                    drawHorizontalAt(val, yAxis, "EF");
                }
            }
            ctx.restore();
        }

        // 4. Draw Crop Box (if in crop mode and this is the active view)
        if (this.mode === 'crop' && this.cropBox) {
            const plane = canvas.dataset?.plane || '2d';
            const isActiveView = (this.ndim === 2 && this.activeCropView === '2d') ||
                (this.ndim === 3 && this.activeCropView === plane);

            if (isActiveView) {
                this.cropBox.render(ctx, displayWidth, displayHeight, MARGIN);
            }
        }
    }

    // Draw EDC and MDC profiles for 2D data
    drawProfiles2D() {
        if (!this.edcCanvas || !this.mdcCanvas || !this.data) return;

        const width = this.shape[1];  // Angle dimension
        const height = this.shape[0]; // Energy dimension

        // Get cursor position from sliceIndices (default to center)
        const cursorX = (this.sliceIndices && this.sliceIndices[1] !== undefined)
            ? this.sliceIndices[1] : Math.floor(width / 2);
        const cursorY = (this.sliceIndices && this.sliceIndices[0] !== undefined)
            ? this.sliceIndices[0] : Math.floor(height / 2);

        // Extract EDC (vertical cut at cursor x position)
        const edc = new Float32Array(height);
        for (let i = 0; i < height; i++) {
            edc[i] = this.data[i * width + Math.min(cursorX, width - 1)];
        }

        // Extract MDC (horizontal cut at cursor y position)
        const mdc = new Float32Array(width);
        const rowStart = Math.min(cursorY, height - 1) * width;
        for (let j = 0; j < width; j++) {
            mdc[j] = this.data[rowStart + j];
        }

        // Draw EDC - cursor is at cursorY (energy index)
        this.drawProfile(this.edcCanvas, edc, this.axes.energy, 'Energy (eV)', '#4caf50', cursorY);

        // Draw MDC - cursor is at cursorX (angle index)
        this.drawProfile(this.mdcCanvas, mdc, this.axes.kx, 'Angle (°)', '#ff9800', cursorX);
    }

    // Draw profiles for 3D data
    drawProfiles3D(sliceXY, idxX, idxY, idxZ) {
        if (!this.edcCanvas || !this.mdcCanvas || !sliceXY) return;

        const width = this.shape[1];  // Angle dimension
        const height = this.shape[0]; // Energy dimension
        const depth = this.shape[2];  // Scan dimension

        // Extract EDC from XY slice (vertical cut at angle position)
        const edc = new Float32Array(height);
        for (let i = 0; i < height; i++) {
            edc[i] = sliceXY[i * width + idxX];
        }

        // Extract MDC from XY slice (horizontal cut at energy position)
        const mdc = new Float32Array(width);
        const rowStart = idxY * width;
        for (let j = 0; j < width; j++) {
            mdc[j] = sliceXY[rowStart + j];
        }

        // Extract Scan profile (at current angle and energy)
        const scanProfile = new Float32Array(depth);
        for (let k = 0; k < depth; k++) {
            // data[idxY, idxX, k] in flattened format: idxY * (width * depth) + idxX * depth + k
            const idx = idxY * (width * depth) + idxX * depth + k;
            scanProfile[k] = this.data[idx];
        }

        // Draw profiles with cursor indicators
        this.drawProfile(this.edcCanvas, edc, this.axes.energy, 'E', '#4caf50', idxY);
        this.drawProfile(this.mdcCanvas, mdc, this.axes.kx, 'θ', '#ff9800', idxX);
        if (this.scanCanvas) {
            this.drawProfile(this.scanCanvas, scanProfile, this.axes.ky, 'Scan', '#2196f3', idxZ);
        }
    }

    // Generic profile drawing method with optional cursor indicator
    drawProfile(canvas, data, axis, label, color, cursorIndex = null) {
        if (!canvas || !data || data.length < 2) return; // Need at least 2 points to draw

        const rect = canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, rect.width, rect.height);

        const padding = PROFILE_PADDING;
        const plotW = rect.width - padding.left - padding.right;
        const plotH = rect.height - padding.top - padding.bottom;

        // Find min/max for scaling
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < data.length; i++) {
            if (isFinite(data[i])) {
                minVal = Math.min(minVal, data[i]);
                maxVal = Math.max(maxVal, data[i]);
            }
        }
        if (!isFinite(minVal) || !isFinite(maxVal) || minVal === maxVal) {
            minVal = 0;
            maxVal = 1;
        }

        // Draw background
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(padding.left, padding.top, plotW, plotH);

        // Draw cursor indicator line (before profile so profile is on top)
        if (cursorIndex !== null && cursorIndex >= 0 && cursorIndex < data.length) {
            const cursorX = padding.left + (cursorIndex / (data.length - 1)) * plotW;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(cursorX, padding.top);
            ctx.lineTo(cursorX, padding.top + plotH);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw profile line
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        const dataLen = data.length - 1; // Safe to divide since we checked length >= 2 above
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / dataLen) * plotW;
            const y = padding.top + plotH - ((data[i] - minVal) / (maxVal - minVal)) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw X-axis ticks (using physical axis values)
        if (axis && axis.length > 1) {
            const axisMin = axis[0];
            const axisMax = axis[axis.length - 1];
            const ticks = this.getNiceTicks(axisMin, axisMax, Math.max(3, Math.floor(plotW / 50)));

            ctx.strokeStyle = '#555';
            ctx.fillStyle = '#888';
            ctx.font = '9px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.lineWidth = 1;

            ticks.forEach(val => {
                const t = (val - axisMin) / (axisMax - axisMin);
                if (t >= 0 && t <= 1) {
                    const x = padding.left + t * plotW;

                    // Draw tick mark
                    ctx.beginPath();
                    ctx.moveTo(x, padding.top + plotH);
                    ctx.lineTo(x, padding.top + plotH + 3);
                    ctx.stroke();

                    // Draw tick label
                    let tickLabel;
                    if (Math.abs(val) < 1e-10) {
                        tickLabel = "0";
                    } else if (Math.abs(val) < 0.01 || Math.abs(val) >= 100) {
                        tickLabel = val.toExponential(0);
                    } else {
                        tickLabel = parseFloat(val.toFixed(1)).toString();
                    }
                    ctx.fillText(tickLabel, x, padding.top + plotH + 4);
                }
            });
        }

        // Draw axis label
        ctx.fillStyle = '#aaa';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(label, padding.left + plotW / 2, rect.height - 3);
    }

    handleInteraction(e, canvas) {
        // Route to crop mode handler if in crop mode
        if (this.mode === 'crop') {
            this.handleCropInteraction(e, canvas);
            return;
        }

        // Route to calibration handler
        if (this.calibrationMode !== 'none') {
            this.handleCalibrationInteraction(e, canvas);
            return;
        }

        // Normal mode: crosshair interaction
        if (e.buttons !== 1 && e.type !== 'mousedown') return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Canvas is now 1:1 with rect (in CSS pixels)
        const plotWidth = rect.width - MARGIN.left - MARGIN.right;
        const plotHeight = rect.height - MARGIN.top - MARGIN.bottom;

        // Check bounds
        if (x < MARGIN.left || x > rect.width - MARGIN.right ||
            y < MARGIN.top || y > rect.height - MARGIN.bottom) {
            return;
        }

        // Map to data indices
        const tX = (x - MARGIN.left) / plotWidth;
        const tY = (rect.height - MARGIN.bottom - y) / plotHeight;

        // Determine data dimensions based on plane
        let dataWidth, dataHeight;
        const plane = canvas.dataset.plane;

        if (this.ndim === 2) {
            dataWidth = this.shape[1];
            dataHeight = this.shape[0];
        } else {
            // 3D
            if (plane === 'xy') {
                dataWidth = this.shape[1]; // Angle
                dataHeight = this.shape[0]; // Energy
            } else if (plane === 'xz') {
                dataWidth = this.shape[2]; // Scan
                dataHeight = this.shape[0]; // Energy
            } else if (plane === 'yz') {
                dataWidth = this.shape[2]; // Scan
                dataHeight = this.shape[1]; // Angle
            }
        }

        let dataX = Math.floor(tX * (dataWidth - 1));
        let dataY = Math.floor(tY * (dataHeight - 1));

        // Clamp
        dataX = Math.max(0, Math.min(dataWidth - 1, dataX));
        dataY = Math.max(0, Math.min(dataHeight - 1, dataY));

        // Update slices
        if (plane === 'xy') {
            this.sliceIndices[1] = dataX; // Angle
            this.sliceIndices[0] = dataY; // Energy
        } else if (plane === 'xz') {
            this.sliceIndices[2] = dataX; // Scan
            this.sliceIndices[0] = dataY; // Energy
        } else if (plane === 'yz') {
            this.sliceIndices[2] = dataX; // Scan
            this.sliceIndices[1] = dataY; // Angle
        } else if (this.ndim === 2) {
            // Just update cursor for 2D
            this.sliceIndices[1] = dataX; // Angle (kx)
            this.sliceIndices[0] = dataY; // Energy
        }

        this.draw();

        const event = new CustomEvent('cursor-update', {
            detail: {
                x: this.sliceIndices[1],
                y: this.sliceIndices[0],
                z: this.sliceIndices[2],
                val: 0 // We could lookup value but it's complex with slices
            }
        });
        document.dispatchEvent(event);
    }

    handleCropInteraction(e, canvas) {
        // Only handle crop box on the active view
        const plane = canvas.dataset.plane || '2d';
        if (this.ndim === 3 && plane !== this.activeCropView) return;
        if (!this.cropBox) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (e.type === 'mousedown') {
            const handle = this.cropBox.hitTest(x, y);
            if (handle) {
                this.cropBox.startDrag(handle, x, y);
                canvas.style.cursor = this.cropBox.getCursor(handle);
            }
        } else if (e.type === 'mousemove') {
            if (this.cropBox.isDragging) {
                this.cropBox.updateDrag(x, y);
                // Constrain to plot area
                const plotRight = rect.width - MARGIN.right;
                const plotBottom = rect.height - MARGIN.bottom;
                this.cropBox.constrain(MARGIN.left, MARGIN.top, plotRight, plotBottom);
                this.draw();
            } else {
                // Update cursor based on hover
                const handle = this.cropBox.hitTest(x, y);
                canvas.style.cursor = handle ? this.cropBox.getCursor(handle) : 'default';
            }
        } else if (e.type === 'mouseup') {
            if (this.cropBox.isDragging) {
                this.cropBox.endDrag();
                canvas.style.cursor = 'default';
            }
        }
    }

    handleCalibrationInteraction(e, canvas) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const plotWidth = rect.width - MARGIN.left - MARGIN.right;
        const plotHeight = rect.height - MARGIN.top - MARGIN.bottom;

        // Determine which axes & labels apply for this view (match draw())
        const plane = canvas.dataset?.plane || '2d';
        let xAxisArr, yAxisArr, xLabel, yLabel;

        if (this.ndim === 2 || plane === 'xy') {
            xAxisArr = this.axes.kx;
            yAxisArr = this.axes.energy;
            xLabel = this.axisLabels.x || "Angle";
            yLabel = this.axisLabels.y || "Energy (eV)";
        } else if (plane === 'yz') {
            xAxisArr = this.axes.ky;
            yAxisArr = this.axes.kx;
            xLabel = this.axisLabels.z || "Scan";
            yLabel = this.axisLabels.x || "Angle";
        } else if (plane === 'xz') {
            xAxisArr = this.axes.ky;
            yAxisArr = this.axes.energy;
            xLabel = this.axisLabels.z || "Scan";
            yLabel = this.axisLabels.y || "Energy (eV)";
        } else {
            // Fallback to global
            xAxisArr = this.axes.kx;
            yAxisArr = this.axes.energy;
            xLabel = this.axisLabels.x || "Angle";
            yLabel = this.axisLabels.y || "Energy (eV)";
        }

        if (!xAxisArr || !yAxisArr) return;

        const xMin = xAxisArr[0];
        const xMax = xAxisArr[xAxisArr.length - 1];
        const yMin = yAxisArr[0];
        const yMax = yAxisArr[yAxisArr.length - 1];

        const lx = (xLabel || '').toLowerCase();
        const ly = (yLabel || '').toLowerCase();
        const xIsAngle = lx.includes('angle') || lx.includes('deg');
        const yIsAngle = ly.includes('angle') || ly.includes('deg');
        const xIsEnergy = lx.includes('energy') || lx.includes('ev');
        const yIsEnergy = ly.includes('energy') || ly.includes('ev');

        const hitThreshold = 10;

        const getAxisKey = (axisArr) => {
            if (axisArr === this.axes.kx) return 'kx';
            if (axisArr === this.axes.ky) return 'ky';
            if (axisArr === this.axes.energy) return 'energy';
            return null;
        };

        const getCalForAxis = (axisArr, fallback) => {
            const k = getAxisKey(axisArr);
            if (k && this.calibrationPositions[k] !== undefined) return this.calibrationPositions[k];
            return fallback;
        };

        const setCalForAxis = (axisArr, val) => {
            const k = getAxisKey(axisArr);
            if (k) {
                this.calibrationPositions[k] = val;
            } else {
                this.calibrationLinePos = val;
            }
        };

        const verticalLineX = (val) => {
            const t = (val - xMin) / (xMax - xMin);
            return MARGIN.left + t * plotWidth;
        };
        const horizontalLineY = (val) => {
            const t = (val - yMin) / (yMax - yMin);
            return MARGIN.top + plotHeight - t * plotHeight;
        };

        if (e.type === 'mousedown') {
            // Determine nearest handle (could be vertical or horizontal)
            let nearest = { axis: null, dist: Infinity };

            if (this.calibrationMode === 'angle') {
                if (xIsAngle) {
                    const lxPos = verticalLineX(getCalForAxis(xAxisArr, this.calibrationLinePos));
                    const d = Math.abs(x - lxPos);
                    if (d < nearest.dist) nearest = { axis: 'x', dist: d };
                }
                if (yIsAngle) {
                    const lyPos = horizontalLineY(getCalForAxis(yAxisArr, this.calibrationLinePosY));
                    const d = Math.abs(y - lyPos);
                    if (d < nearest.dist) nearest = { axis: 'y', dist: d };
                }
            } else if (this.calibrationMode === 'fermi') {
                if (xIsEnergy) {
                    const lxPos = verticalLineX(getCalForAxis(xAxisArr, this.calibrationLinePos));
                    const d = Math.abs(x - lxPos);
                    if (d < nearest.dist) nearest = { axis: 'x', dist: d };
                }
                if (yIsEnergy) {
                    const lyPos = horizontalLineY(getCalForAxis(yAxisArr, this.calibrationLinePos));
                    const d = Math.abs(y - lyPos);
                    if (d < nearest.dist) nearest = { axis: 'y', dist: d };
                }
            }

            if (nearest.axis && nearest.dist < hitThreshold) {
                this.isDraggingCalibration = true;
                this.calibrationDragAxis = nearest.axis;
                this.calibrationDragAxisArr = (nearest.axis === 'x') ? xAxisArr : yAxisArr;
                this.calibrationDragAxisKey = getAxisKey(this.calibrationDragAxisArr);
                canvas.style.cursor = (nearest.axis === 'x') ? 'ew-resize' : 'ns-resize';
            }
        } else if (e.type === 'mousemove') {
            if (this.isDraggingCalibration) {
                if (this.calibrationDragAxis === 'x') {
                    let t = (x - margin.left) / plotWidth;
                    t = Math.max(0, Math.min(1, t));
                    const newVal = xMin + t * (xMax - xMin);
                    // Set calibration for the specific axis being dragged
                    setCalForAxis(this.calibrationDragAxisArr, newVal);
                } else if (this.calibrationDragAxis === 'y') {
                    let t = (rect.height - margin.bottom - y) / plotHeight;
                    t = Math.max(0, Math.min(1, t));
                    const newVal = yMin + t * (yMax - yMin);
                    setCalForAxis(this.calibrationDragAxisArr, newVal);
                }
                this.draw();
            } else {
                // Hover cursor detection for both possible lines
                let nearestCursor = 'default';
                let nearestDist = Infinity;

                if (this.calibrationMode === 'angle') {
                    if (xIsAngle) {
                        const lxPos = verticalLineX(getCalForAxis(xAxisArr, this.calibrationLinePos));
                        const d = Math.abs(x - lxPos);
                        if (d < nearestDist) {
                            nearestDist = d;
                            nearestCursor = 'ew-resize';
                        }
                    }
                    if (yIsAngle) {
                        const lyPos = horizontalLineY(getCalForAxis(yAxisArr, this.calibrationLinePosY));
                        const d = Math.abs(y - lyPos);
                        if (d < nearestDist) {
                            nearestDist = d;
                            nearestCursor = 'ns-resize';
                        }
                    }
                } else if (this.calibrationMode === 'fermi') {
                    if (xIsEnergy) {
                        const lxPos = verticalLineX(getCalForAxis(xAxisArr, this.calibrationLinePos));
                        const d = Math.abs(x - lxPos);
                        if (d < nearestDist) {
                            nearestDist = d;
                            nearestCursor = 'ew-resize';
                        }
                    }
                    if (yIsEnergy) {
                        const lyPos = horizontalLineY(getCalForAxis(yAxisArr, this.calibrationLinePos));
                        const d = Math.abs(y - lyPos);
                        if (d < nearestDist) {
                            nearestDist = d;
                            nearestCursor = 'ns-resize';
                        }
                    }
                }

                if (nearestDist < hitThreshold) {
                    canvas.style.cursor = nearestCursor;
                } else {
                    canvas.style.cursor = 'default';
                }
            }
        } else if (e.type === 'mouseup') {
            this.isDraggingCalibration = false;
            this.calibrationDragAxis = null;
            canvas.style.cursor = 'default';
        }
    }

    handleViewClick(e, canvas) {
        // Handle view selection in crop mode for 3D
        if (this.mode !== 'crop') return;
        if (this.ndim !== 3) return;
        if (this.activeCropView) return; // Already selected

        const plane = canvas.dataset.plane;
        if (plane) {
            this.selectCropView(plane);
        }
    }

    /**
     * Setup mouse event listeners for a profile canvas
     * @param {HTMLCanvasElement} canvas - The profile canvas element
     * @param {string} profileType - 'edc', 'mdc', or 'scan'
     */
    setupProfileInteraction(canvas, profileType) {
        if (!canvas) return;

        canvas.style.cursor = 'crosshair';

        const handler = (e) => this.handleProfileInteraction(e, canvas, profileType);
        canvas.addEventListener('mousedown', handler);
        canvas.addEventListener('mousemove', handler);
        canvas.addEventListener('mouseup', handler);
    }

    /**
     * Handle mouse interaction on profile canvases to update cursor position
     * @param {MouseEvent} e - The mouse event
     * @param {HTMLCanvasElement} canvas - The profile canvas
     * @param {string} profileType - 'edc', 'mdc', or 'scan'
     */
    handleProfileInteraction(e, canvas, profileType) {
        // Only respond to left-click drag or mousedown
        if (e.buttons !== 1 && e.type !== 'mousedown') return;

        const rect = canvas.parentElement.getBoundingClientRect();
        const x = e.clientX - rect.left;

        // Profile padding (must match drawProfile)
        const padding = PROFILE_PADDING;
        const plotW = rect.width - padding.left - padding.right;

        // Check if click is within plot area
        if (x < padding.left || x > rect.width - padding.right) return;

        // Convert x position to normalized value (0-1)
        const t = (x - padding.left) / plotW;

        // Get the data length for this profile type
        let dataLength;
        if (this.ndim === 2) {
            if (profileType === 'edc') {
                dataLength = this.shape[0]; // Energy dimension
            } else if (profileType === 'mdc') {
                dataLength = this.shape[1]; // Angle dimension
            }
        } else if (this.ndim === 3) {
            if (profileType === 'edc') {
                dataLength = this.shape[0]; // Energy dimension
            } else if (profileType === 'mdc') {
                dataLength = this.shape[1]; // Angle dimension
            } else if (profileType === 'scan') {
                dataLength = this.shape[2]; // Scan dimension
            }
        }

        if (!dataLength) return;

        // Convert to data index
        let dataIndex = Math.round(t * (dataLength - 1));
        dataIndex = Math.max(0, Math.min(dataLength - 1, dataIndex));

        // Update the appropriate sliceIndices based on profile type
        // EDC: x-axis is Energy, so clicking sets energy index (sliceIndices[0])
        // MDC: x-axis is Angle, so clicking sets angle index (sliceIndices[1])
        // Scan: x-axis is Scan, so clicking sets scan index (sliceIndices[2])
        if (profileType === 'edc') {
            this.sliceIndices[0] = dataIndex; // Energy index
        } else if (profileType === 'mdc') {
            this.sliceIndices[1] = dataIndex; // Angle index
        } else if (profileType === 'scan') {
            this.sliceIndices[2] = dataIndex; // Scan index
        }

        // Redraw to update crosshairs and profiles
        this.draw();

        // Dispatch cursor-update event for status updates
        const event = new CustomEvent('cursor-update', {
            detail: {
                x: this.sliceIndices[1],
                y: this.sliceIndices[0],
                z: this.sliceIndices[2],
                val: 0
            }
        });
        document.dispatchEvent(event);
    }

    onResize() {
        // Handle resize if needed
        this.draw();
    }

    setColormap(name) {
        this.colormap = name;
        this.currentLut = this.getColormap(name);
        this.draw();
    }

    setContrastMin(val) {
        this.contrastMin = val;
        this.draw();
    }

    setContrastMax(val) {
        this.contrastMax = val;
        this.draw();
    }

    setAxisLabels(labels) {
        this.axisLabels = { ...this.axisLabels, ...labels };
        this.draw();
    }

    getContrastRange() {
        return { min: this.contrastMin, max: this.contrastMax };
    }

    autoContrast() {
        if (!this.data || !this.metadata) return;

        const globalMin = this.metadata.data_info.min;
        const globalMax = this.metadata.data_info.max;
        const range = globalMax - globalMin;

        if (range <= 0) {
            this.contrastMin = 0;
            this.contrastMax = 100;
            this.draw();
            return;
        }

        // Percentile based auto-contrast (histogram approach)
        // We use a simplified histogram with 1000 bins to find ~2nd and ~98th percentiles.
        const numBins = 1000;
        const histogram = new Uint32Array(numBins);
        const data = this.data;
        const len = data.length;

        // Sample data for speed if dataset is huge (>1M points)
        // Stride = 1 if len < 1M, else skip appropriately
        const stride = len > 1000000 ? Math.floor(len / 1000000) : 1;

        let sampleCount = 0;
        for (let i = 0; i < len; i += stride) {
            const val = data[i];
            // Normalize to 0-1 relative to global range
            let t = (val - globalMin) / range;
            if (t < 0) t = 0;
            if (t >= 1) t = 0.9999; // Ensure it fits in the last bin

            const bin = Math.floor(t * numBins);
            histogram[bin]++;
            sampleCount++;
        }

        // Find lower cut (2nd percentile)
        const lowCutThreshold = sampleCount * 0.02;
        let count = 0;
        let lowCutBin = 0;
        for (let i = 0; i < numBins; i++) {
            count += histogram[i];
            if (count > lowCutThreshold) {
                lowCutBin = i;
                break;
            }
        }

        // Find upper cut (98th percentile)
        const highCutThreshold = sampleCount * 0.98;
        count = 0;
        let highCutBin = numBins - 1;
        for (let i = 0; i < numBins; i++) {
            count += histogram[i];
            if (count > highCutThreshold) {
                highCutBin = i;
                break;
            }
        }

        // Convert bin indices back to percent range strings (0-100)
        // lowCutBin / numBins is 0-1 float.
        this.contrastMin = (lowCutBin / numBins) * 100;
        this.contrastMax = (highCutBin / numBins) * 100;

        // Safety check to ensure they don't cross or get stuck
        if (this.contrastMax <= this.contrastMin) {
            this.contrastMin = 0;
            this.contrastMax = 100;
        }

        if (DEBUG) console.log(`Auto Contrast: Min=${this.contrastMin.toFixed(2)}%, Max=${this.contrastMax.toFixed(2)}%`);

        this.draw();
    }

    resetContrast() {
        this.contrastMin = 0;
        this.contrastMax = 100;
        this.draw();
    }

    setInverted(inverted) {
        this.inverted = inverted;
        this.currentLut = this.getColormap(this.colormap);
        this.draw();
    }

    getColormap(name) {
        // Check cache first (include inversion state in cache key)
        const cacheKey = `${name}_${this.inverted ? 'inv' : 'normal'}`;
        if (this.colormapCache[cacheKey]) {
            return this.colormapCache[cacheKey];
        }

        // Generate LUT
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 1;
        const ctx = canvas.getContext('2d');
        const grd = ctx.createLinearGradient(0, 0, 256, 0);

        if (name === 'viridis') {
            grd.addColorStop(0, '#440154');
            grd.addColorStop(0.25, '#3b528b');
            grd.addColorStop(0.5, '#21918c');
            grd.addColorStop(0.75, '#5ec962');
            grd.addColorStop(1, '#fde725');
        } else if (name === 'plasma') {
            grd.addColorStop(0, '#0d0887');
            grd.addColorStop(0.25, '#7e03a8');
            grd.addColorStop(0.5, '#cc4778');
            grd.addColorStop(0.75, '#f89540');
            grd.addColorStop(1, '#f0f921');
        } else if (name === 'inferno') {
            grd.addColorStop(0, '#000004');
            grd.addColorStop(0.25, '#420a68');
            grd.addColorStop(0.5, '#932667');
            grd.addColorStop(0.75, '#dd513a');
            grd.addColorStop(1, '#fcffa4');
        } else if (name === 'magma') {
            grd.addColorStop(0, '#000004');
            grd.addColorStop(0.25, '#3b0f70');
            grd.addColorStop(0.5, '#8c2981');
            grd.addColorStop(0.75, '#fe9f6d');
            grd.addColorStop(1, '#fcfdbf');
        } else if (name === 'cividis') {
            grd.addColorStop(0, '#00204c');
            grd.addColorStop(0.5, '#7c7b78');
            grd.addColorStop(1, '#ffe945');
        } else if (name === 'hot') {
            grd.addColorStop(0, 'black');
            grd.addColorStop(0.33, 'red');
            grd.addColorStop(0.66, 'yellow');
            grd.addColorStop(1, 'white');
        } else if (name === 'cool') {
            grd.addColorStop(0, 'cyan');
            grd.addColorStop(1, 'magenta');
        } else if (name === 'rdbu') {
            grd.addColorStop(0, 'red');
            grd.addColorStop(0.5, 'white');
            grd.addColorStop(1, 'blue');
        } else if (name === 'seismic') {
            grd.addColorStop(0, 'blue');
            grd.addColorStop(0.5, 'white');
            grd.addColorStop(1, 'red');
        } else if (name === 'gray') {
            grd.addColorStop(0, 'black');
            grd.addColorStop(1, 'white');
        } else if (name === 'bone') {
            // Bone colormap: blue-tinted grayscale
            grd.addColorStop(0, '#000000');
            grd.addColorStop(0.25, '#2a3a4a');
            grd.addColorStop(0.5, '#6a8090');
            grd.addColorStop(0.75, '#b0c8d8');
            grd.addColorStop(1, '#ffffff');
        } else {
            // Default fallback
            grd.addColorStop(0, 'black');
            grd.addColorStop(1, 'white');
        }

        ctx.fillStyle = grd;
        ctx.fillRect(0, 0, 256, 1);

        const imgData = ctx.getImageData(0, 0, 256, 1);
        const data = imgData.data;
        let resultData;

        // Handle Inversion
        if (this.inverted) {
            const invertedData = new Uint8ClampedArray(data.length);
            for (let i = 0; i < 256; i++) {
                const srcIdx = i * 4;
                const dstIdx = (255 - i) * 4;
                invertedData[dstIdx] = data[srcIdx];
                invertedData[dstIdx + 1] = data[srcIdx + 1];
                invertedData[dstIdx + 2] = data[srcIdx + 2];
                invertedData[dstIdx + 3] = data[srcIdx + 3];
            }
            resultData = invertedData;
        } else {
            resultData = data;
        }

        // Cache and return
        this.colormapCache[cacheKey] = resultData;
        return resultData;
    }
    getNiceTicks(min, max, maxTicks = 5) {
        const rangeVal = max - min;
        // Handle edge cases to avoid NaN
        if (!isFinite(rangeVal) || rangeVal <= 0) {
            return [min];
        }
        const range = this.niceNum(rangeVal, false);
        const tickSpacing = this.niceNum(range / (maxTicks - 1), true);

        if (!isFinite(tickSpacing) || tickSpacing <= 0) {
            return [min, max];
        }

        const niceMin = Math.ceil(min / tickSpacing) * tickSpacing;
        const niceMax = Math.floor(max / tickSpacing) * tickSpacing;

        const ticks = [];
        for (let t = niceMin; t <= niceMax + 0.5 * tickSpacing; t += tickSpacing) {
            ticks.push(t);
        }
        return ticks;
    }

    niceNum(range, round) {
        // Handle edge cases
        if (!isFinite(range) || range <= 0) {
            return 1;
        }

        const exponent = Math.floor(Math.log10(range));
        const fraction = range / Math.pow(10, exponent);
        let niceFraction;

        if (round) {
            if (fraction < 1.5) niceFraction = 1;
            else if (fraction < 3) niceFraction = 2;
            else if (fraction < 7) niceFraction = 5;
            else niceFraction = 10;
        } else {
            if (fraction <= 1) niceFraction = 1;
            else if (fraction <= 2) niceFraction = 2;
            else if (fraction <= 5) niceFraction = 5;
            else niceFraction = 10;
        }

        return niceFraction * Math.pow(10, exponent);
    }
    extractSlice(dim, index) {
        if (!this.data || this.ndim !== 3) return null;

        const [d0, d1, d2] = this.shape;
        // Data is flat: i0 * (d1*d2) + i1 * d2 + i2

        if (dim === 0) {
            // Fix dim 0 (e.g. Energy). Slice is (d1, d2).
            // Contiguous block if d0 is slowest varying? 
            // Usually row-major: d0 is slowest.
            // data[index, :, :]
            // Start index: index * (d1*d2)
            // Length: d1*d2
            const size = d1 * d2;
            const start = index * size;
            return this.data.slice(start, start + size);
        } else if (dim === 1) {
            // Fix dim 1. Slice is (d0, d2).
            // data[:, index, :]
            // For each i0 in 0..d0-1:
            //   copy row data[i0, index, :] (length d2)
            const size = d0 * d2;
            const slice = new Float32Array(size);
            let ptr = 0;
            for (let i0 = 0; i0 < d0; i0++) {
                const rowStart = i0 * (d1 * d2) + index * d2;
                for (let i2 = 0; i2 < d2; i2++) {
                    slice[ptr++] = this.data[rowStart + i2];
                }
            }
            return slice;
        } else if (dim === 2) {
            // Fix dim 2. Slice is (d0, d1).
            // data[:, :, index]
            // Pick one element from each row of d2 elements.
            const size = d0 * d1;
            const slice = new Float32Array(size);
            let ptr = 0;
            for (let i0 = 0; i0 < d0; i0++) {
                for (let i1 = 0; i1 < d1; i1++) {
                    const idx = i0 * (d1 * d2) + i1 * d2 + index;
                    slice[ptr++] = this.data[idx];
                }
            }
            return slice;
        }
        return null;
    }

    exportImage() {
        if (!this.data) return;

        let exportCanvas;

        if (this.ndim === 2) {
            // 2D: Just use the single canvas
            exportCanvas = this.canvases[0];
        } else if (this.ndim === 3) {
            // 3D: Composite the 3 views
            // Layout: 
            // [XY] [XZ]
            // [YZ] [Info]
            // We'll create a canvas that fits them all.
            // Assuming grid layout with 2 columns.

            const c1 = this.canvases[0]; // XY
            const c2 = this.canvases[1]; // XZ
            const c3 = this.canvases[2]; // YZ

            const width = c1.width + c2.width;
            const height = c1.height + c3.height; // Approximate

            exportCanvas = document.createElement('canvas');
            exportCanvas.width = width;
            exportCanvas.height = height;
            const ctx = exportCanvas.getContext('2d');

            // Fill white background
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, width, height);

            // Draw canvases
            // XY (Top Left)
            ctx.drawImage(c1, 0, 0);
            // XZ (Top Right)
            ctx.drawImage(c2, c1.width, 0);
            // YZ (Bottom Left)
            ctx.drawImage(c3, 0, c1.height);

            // Add some text?
            ctx.fillStyle = 'black';
            ctx.font = '20px sans-serif';
            ctx.fillText("3D View Composite", c1.width + 20, c1.height + 40);
        }

        // Trigger Download
        const link = document.createElement('a');
        link.download = 'arpes_data_export.png';
        link.href = exportCanvas.toDataURL('image/png');
        link.click();
    }

    // ========== CROP MODE METHODS ==========

    enterCropMode() {
        if (!this.data) {
            console.warn("No data loaded, cannot enter crop mode");
            return false;
        }

        // Backup current data
        this.originalDataBackup = this.data.slice();
        this.originalMetaBackup = JSON.parse(JSON.stringify(this.metadata));

        this.mode = 'crop';
        this.activeCropView = null;
        this.cropBox = null;

        // For 2D, automatically select the view and create crop box
        if (this.ndim === 2) {
            this.activeCropView = '2d';
            this.createDefaultCropBox(this.canvases[0]);
        }

        // Dispatch event to notify app
        document.dispatchEvent(new CustomEvent('crop-mode-entered'));

        this.draw();
        return true;
    }

    exitCropMode() {
        this.mode = 'normal';
        this.activeCropView = null;
        this.cropBox = null;
        this.originalDataBackup = null;
        this.originalMetaBackup = null;

        // Remove active view highlight
        this.canvases.forEach(canvas => {
            canvas.parentElement.classList.remove('crop-active-view');
        });

        // Dispatch event to notify app
        document.dispatchEvent(new CustomEvent('crop-mode-exited'));

        this.draw();
    }

    selectCropView(plane) {
        if (this.mode !== 'crop') return;
        if (this.ndim !== 3) return;

        this.activeCropView = plane;

        // Find the canvas for this plane
        const canvas = this.canvases.find(c => c.dataset.plane === plane);
        if (canvas) {
            // Remove highlight from all canvases
            this.canvases.forEach(c => {
                c.parentElement.classList.remove('crop-active-view');
            });

            // Add highlight to selected canvas
            canvas.parentElement.classList.add('crop-active-view');

            // Create default crop box
            this.createDefaultCropBox(canvas);
            this.draw();
        }
    }

    createDefaultCropBox(canvas) {
        const rect = canvas.parentElement.getBoundingClientRect();

        const plotWidth = rect.width - MARGIN.left - MARGIN.right;
        const plotHeight = rect.height - MARGIN.top - MARGIN.bottom;

        // Create crop box at 20% inset from edges
        const inset = 0.2;
        const x = MARGIN.left + plotWidth * inset;
        const y = MARGIN.top + plotHeight * inset;
        const width = plotWidth * (1 - 2 * inset);
        const height = plotHeight * (1 - 2 * inset);

        this.cropBox = new CropBox(x, y, width, height);
    }

    getCropIndices() {
        if (!this.cropBox || !this.activeCropView) return null;

        let canvas, dataWidth, dataHeight;

        if (this.ndim === 2) {
            canvas = this.canvases[0];
            dataWidth = this.shape[1];
            dataHeight = this.shape[0];
        } else {
            const plane = this.activeCropView;
            canvas = this.canvases.find(c => c.dataset.plane === plane);

            if (plane === 'xy') {
                dataWidth = this.shape[1]; // Angle
                dataHeight = this.shape[0]; // Energy
            } else if (plane === 'xz') {
                dataWidth = this.shape[2]; // Scan
                dataHeight = this.shape[0]; // Energy
            } else if (plane === 'yz') {
                dataWidth = this.shape[2]; // Scan
                dataHeight = this.shape[1]; // Angle
            }
        }

        const rect = canvas.parentElement.getBoundingClientRect();
        const bounds = this.cropBox.getNormalizedBounds(MARGIN, rect.width, rect.height);

        // Convert normalized bounds to data indices
        // bounds.y0 is at top (normalized), but we need to flip for data coordinates
        const x0 = Math.floor(bounds.x0 * (dataWidth - 1));
        const x1 = Math.ceil(bounds.x1 * (dataWidth - 1));
        const y0 = Math.floor((1 - bounds.y1) * (dataHeight - 1)); // Flip Y
        const y1 = Math.ceil((1 - bounds.y0) * (dataHeight - 1)); // Flip Y

        return {
            x0: Math.max(0, Math.min(x0, dataWidth - 1)),
            x1: Math.max(0, Math.min(x1, dataWidth - 1)),
            y0: Math.max(0, Math.min(y0, dataHeight - 1)),
            y1: Math.max(0, Math.min(y1, dataHeight - 1))
        };
    }

    applyCrop() {
        if (!this.cropBox || !this.activeCropView) {
            console.warn("No crop box defined");
            return false;
        }

        const indices = this.getCropIndices();
        if (!indices) return false;

        if (this.ndim === 2) {
            this.applyCrop2D(indices);
        } else if (this.ndim === 3) {
            this.applyCrop3D(indices);
        }

        this.exitCropMode();
        return true;
    }

    applyCrop2D(indices) {
        const { x0, x1, y0, y1 } = indices;
        const oldWidth = this.shape[1];
        const oldHeight = this.shape[0];

        const newWidth = x1 - x0 + 1;
        const newHeight = y1 - y0 + 1;

        // Crop data
        const newData = new Float32Array(newWidth * newHeight);
        for (let y = 0; y < newHeight; y++) {
            for (let x = 0; x < newWidth; x++) {
                const oldIdx = (y0 + y) * oldWidth + (x0 + x);
                const newIdx = y * newWidth + x;
                newData[newIdx] = this.data[oldIdx];
            }
        }

        // Update axes
        this.axes.kx = this.axes.kx.slice(x0, x1 + 1);
        this.axes.energy = this.axes.energy.slice(y0, y1 + 1);

        // Update metadata
        this.data = newData;
        this.shape = [newHeight, newWidth];
        this.metadata.data_info.shape = this.shape;
        this.metadata.axes = this.axes;

        // Recalculate min/max
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < newData.length; i++) {
            const v = newData[i];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        this.metadata.data_info.min = min;
        this.metadata.data_info.max = max;

        // Reset slice indices
        this.sliceIndices = [Math.floor(newHeight / 2), Math.floor(newWidth / 2), 0];

        this.renderLayout();
        this.draw();
    }

    applyCrop3D(indices) {
        const { x0, x1, y0, y1 } = indices;
        const plane = this.activeCropView;
        const [d0, d1, d2] = this.shape;

        let newShape, newData, newAxes;

        if (plane === 'xy') {
            // Crop dim0 (Energy) and dim1 (Angle), keep dim2 (Scan)
            const newD0 = y1 - y0 + 1;
            const newD1 = x1 - x0 + 1;
            newShape = [newD0, newD1, d2];
            newData = new Float32Array(newD0 * newD1 * d2);

            for (let i0 = 0; i0 < newD0; i0++) {
                for (let i1 = 0; i1 < newD1; i1++) {
                    for (let i2 = 0; i2 < d2; i2++) {
                        const oldIdx = (y0 + i0) * (d1 * d2) + (x0 + i1) * d2 + i2;
                        const newIdx = i0 * (newD1 * d2) + i1 * d2 + i2;
                        newData[newIdx] = this.data[oldIdx];
                    }
                }
            }

            newAxes = {
                energy: this.axes.energy.slice(y0, y1 + 1),
                kx: this.axes.kx.slice(x0, x1 + 1),
                ky: this.axes.ky
            };

        } else if (plane === 'yz') {
            // Crop dim1 (Angle) and dim2 (Scan), keep dim0 (Energy)
            const newD1 = y1 - y0 + 1;
            const newD2 = x1 - x0 + 1;
            newShape = [d0, newD1, newD2];
            newData = new Float32Array(d0 * newD1 * newD2);

            for (let i0 = 0; i0 < d0; i0++) {
                for (let i1 = 0; i1 < newD1; i1++) {
                    for (let i2 = 0; i2 < newD2; i2++) {
                        const oldIdx = i0 * (d1 * d2) + (y0 + i1) * d2 + (x0 + i2);
                        const newIdx = i0 * (newD1 * newD2) + i1 * newD2 + i2;
                        newData[newIdx] = this.data[oldIdx];
                    }
                }
            }

            newAxes = {
                energy: this.axes.energy,
                kx: this.axes.kx.slice(y0, y1 + 1),
                ky: this.axes.ky.slice(x0, x1 + 1)
            };

        } else if (plane === 'xz') {
            // Crop dim0 (Energy) and dim2 (Scan), keep dim1 (Angle)
            const newD0 = y1 - y0 + 1;
            const newD2 = x1 - x0 + 1;
            newShape = [newD0, d1, newD2];
            newData = new Float32Array(newD0 * d1 * newD2);

            for (let i0 = 0; i0 < newD0; i0++) {
                for (let i1 = 0; i1 < d1; i1++) {
                    for (let i2 = 0; i2 < newD2; i2++) {
                        const oldIdx = (y0 + i0) * (d1 * d2) + i1 * d2 + (x0 + i2);
                        const newIdx = i0 * (d1 * newD2) + i1 * newD2 + i2;
                        newData[newIdx] = this.data[oldIdx];
                    }
                }
            }

            newAxes = {
                energy: this.axes.energy.slice(y0, y1 + 1),
                kx: this.axes.kx,
                ky: this.axes.ky.slice(x0, x1 + 1)
            };
        }

        // Update state
        this.data = newData;
        this.shape = newShape;
        this.axes = newAxes;
        this.metadata.data_info.shape = newShape;
        this.metadata.axes = newAxes;

        // Recalculate min/max
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < newData.length; i++) {
            const v = newData[i];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        this.metadata.data_info.min = min;
        this.metadata.data_info.max = max;

        // Reset slice indices to center
        this.sliceIndices = newShape.map(d => Math.floor(d / 2));

        this.renderLayout();
        this.draw();
    }

    cancelCrop() {
        if (this.originalDataBackup && this.originalMetaBackup) {
            this.data = this.originalDataBackup;
            this.metadata = this.originalMetaBackup;
            this.axes = this.originalMetaBackup.axes;
            this.shape = this.originalMetaBackup.data_info.shape;
        }

        this.exitCropMode();
    }

    // ========== CALIBRATION MODE METHODS ==========

    enterCalibrationMode(type) {
        if (!this.data) return false;
        this.calibrationMode = type;

        // Initialize line position to center based on currently displayed axes
        if (type === 'angle') {
            // Initialize X position (kx axis)
            if (this.axes.kx && this.axes.kx.length > 0) {
                const xMin = this.axes.kx[0];
                const xMax = this.axes.kx[this.axes.kx.length - 1];
                this.calibrationLinePos = (xMin + xMax) / 2;
            } else {
                this.calibrationLinePos = 0;
            }

            // Initialize Y position (ky axis for 3D data)
            if (this.axes.ky && this.axes.ky.length > 0) {
                const yMin = this.axes.ky[0];
                const yMax = this.axes.ky[this.axes.ky.length - 1];
                this.calibrationLinePosY = (yMin + yMax) / 2;
            } else {
                this.calibrationLinePosY = this.calibrationLinePos;
            }
        } else if (type === 'fermi') {
            // Horizontal line (Y-axis)
            const yMin = this.axes.energy[0];
            const yMax = this.axes.energy[this.axes.energy.length - 1];
            this.calibrationLinePos = (yMin + yMax) / 2;
        }

        this.draw();
        return true;
    }

    exitCalibrationMode() {
        this.calibrationMode = 'none';
        this.isDraggingCalibration = false;
        this.draw();
    }

    getCalibrationValue() {
        // Helper to get the current calibration value based on mode and active axes
        // This mirrors the logic in handleCalibrationInteraction/draw

        // Default fallback
        let val = this.calibrationLinePos;

        // Try to find the specific axis value if possible
        if (this.calibrationMode === 'angle') {
            // For angle, we usually care about kx or ky
            // If we have a specific updated value in calibrationPositions, use it.
            // But which one? The UI is ambiguous if both X and Y are angles (rare in 2D, possible in 3D).
            // For 2D, it's usually kx.
            if (this.calibrationPositions.kx !== undefined) val = this.calibrationPositions.kx;
            else if (this.calibrationPositions.ky !== undefined) val = this.calibrationPositions.ky;
        } else if (this.calibrationMode === 'fermi') {
            // For fermi, it's energy.
            if (this.calibrationPositions.energy !== undefined) val = this.calibrationPositions.energy;
        }

        return val;
    }

}
