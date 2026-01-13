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

        // Cursor Width State (for profile integration)
        this.cursorWidth = {
            energy: 0,  // eV (converted to pixels using axis)
            angle: 0,   // pixels
            scan: 0     // pixels
        };
    }

    getThemeColor(varName, fallback) {
        const val = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
        return val || fallback;
    }

    setEnhancement(opts) {
        this.enhancement = { ...this.enhancement, ...opts };
        this.draw();
    }

    /**
     * Set cursor width for profile integration.
     * @param {string} type - 'energy' (eV), 'angle' (px), or 'scan' (px)
     * @param {number} value - Width value
     */
    setCursorWidth(type, value) {
        if (type === 'energy' || type === 'angle' || type === 'scan') {
            this.cursorWidth[type] = value;
            this.draw();
        }
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

            // Get cursor widths for integrated slice extraction
            const scanWidthPx = this.cursorWidth.scan || 0;
            const angleWidthPx = this.cursorWidth.angle || 0;
            const energyWidthEV = this.cursorWidth.energy || 0;

            // Convert energy width from eV to pixels
            let energyWidthPx = 0;
            if (energyWidthEV > 0 && this.axes.energy && this.axes.energy.length > 1) {
                const eMin = this.axes.energy[0];
                const eMax = this.axes.energy[this.axes.energy.length - 1];
                const eRange = Math.abs(eMax - eMin);
                const pxPerEV = this.shape[0] / eRange;
                energyWidthPx = Math.round(energyWidthEV * pxPerEV);
            }

            // 1. XY View (Z-cut): Energy vs Angle - integrate over scan width
            // Overlay shows angle (x) and energy (y) widths
            const sliceXY = this.extractSliceIntegrated(2, idxZ, scanWidthPx);
            this.draw2D(
                this.canvases[0],
                sliceXY,
                this.shape[1], this.shape[0],
                this.axes.kx, this.axes.energy,
                this.axisLabels.x || "Angle", this.axisLabels.y || "Energy (eV)",
                [idxX, idxY],
                { x: angleWidthPx, y: energyWidthPx }
            );

            // 2. XZ View (Y-cut): Energy vs Scan - integrate over angle width
            // X-axis: Scan (dim2), Y-axis: Energy (dim0)
            // Overlay shows scan (x) and energy (y) widths
            const sliceXZ = this.extractSliceIntegrated(1, idxX, angleWidthPx);
            this.draw2D(
                this.canvases[1],
                sliceXZ,
                this.shape[2], this.shape[0],
                this.axes.ky, this.axes.energy,
                this.axisLabels.z || "Scan", this.axisLabels.y || "Energy (eV)",
                [idxZ, idxY],
                { x: scanWidthPx, y: energyWidthPx }
            );

            // 3. YZ View (X-cut): Angle vs Scan - integrate over energy width
            // X-axis: Scan (dim2), Y-axis: Angle (dim1)
            // Overlay shows scan (x) and angle (y) widths
            const sliceYZ = this.extractSliceIntegrated(0, idxY, energyWidthPx);
            this.draw2D(
                this.canvases[2],
                sliceYZ,
                this.shape[2], this.shape[1],
                this.axes.ky, this.axes.kx,
                this.axisLabels.z || "Scan", this.axisLabels.x || "Angle",
                [idxZ, idxX],
                { x: scanWidthPx, y: angleWidthPx }
            );

            // Draw profiles for 3D
            this.drawProfiles3D(sliceXY, idxX, idxY, idxZ);
        }
    }

    draw2D(canvas, data, dataWidth, dataHeight, xAxis, yAxis, xLabel, yLabel, crosshair = [], overlayWidths = null, skipCrosshair = false, exportDimensions = null, exportScale = null) {
        // Get container size - use explicit dimensions for export or getBoundingClientRect for normal rendering
        let displayWidth, displayHeight;
        if (exportDimensions) {
            displayWidth = exportDimensions.width;
            displayHeight = exportDimensions.height;
        } else {
            const rect = canvas.parentElement.getBoundingClientRect();
            displayWidth = rect.width;
            displayHeight = rect.height;
        }

        // For export, use provided scale; otherwise use device pixel ratio
        const dpr = exportScale || window.devicePixelRatio || 1;

        // Set canvas resolution
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
        const axisColor = this.getThemeColor('--text-secondary', '#888');
        const tickColor = this.getThemeColor('--text-secondary', '#ccc'); // Using secondary for ticks too
        const labelColor = this.getThemeColor('--text-primary', '#e0e0e0');

        ctx.strokeStyle = axisColor;
        ctx.fillStyle = tickColor;
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
            ctx.fillStyle = labelColor;
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
            ctx.fillStyle = tickColor;

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
            ctx.fillStyle = labelColor;
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

            // Draw integration region overlay (before crosshair lines)
            // Use overlayWidths if provided (for 3D views), otherwise default to 2D behavior
            let xWidthPx = 0;
            let yWidthPx = 0;

            if (overlayWidths) {
                // 3D views provide explicit widths in pixels
                xWidthPx = overlayWidths.x || 0;
                yWidthPx = overlayWidths.y || 0;
            } else {
                // 2D view: use angle and energy defaults
                xWidthPx = this.cursorWidth.angle || 0;
                const energyWidthEV = this.cursorWidth.energy || 0;
                if (energyWidthEV > 0 && yAxis && yAxis.length > 1) {
                    const eMin = yAxis[0];
                    const eMax = yAxis[yAxis.length - 1];
                    const eRange = Math.abs(eMax - eMin);
                    const pxPerEV = dataHeight / eRange;
                    yWidthPx = Math.round(energyWidthEV * pxPerEV);
                }
            }

            // Calculate overlay bounds in canvas coordinates
            if (xWidthPx > 0 || yWidthPx > 0) {
                const halfWidthX = (xWidthPx / (dataWidth - 1)) * plotWidth;
                const halfWidthY = (yWidthPx / (dataHeight - 1)) * plotHeight;

                const overlayLeft = canvasX - halfWidthX;
                const overlayRight = canvasX + halfWidthX;
                const overlayTop = canvasY - halfWidthY;
                const overlayBottom = canvasY + halfWidthY;

                // Draw vertical band (X-axis integration range)
                if (xWidthPx > 0) {
                    ctx.fillStyle = 'rgba(255, 152, 0, 0.35)'; // Orange
                    ctx.fillRect(
                        Math.max(MARGIN.left, overlayLeft),
                        MARGIN.top,
                        Math.min(overlayRight, MARGIN.left + plotWidth) - Math.max(MARGIN.left, overlayLeft),
                        plotHeight
                    );
                }

                // Draw horizontal band (Y-axis integration range)
                if (yWidthPx > 0) {
                    ctx.fillStyle = 'rgba(76, 175, 80, 0.35)'; // Green
                    ctx.fillRect(
                        MARGIN.left,
                        Math.max(MARGIN.top, overlayTop),
                        plotWidth,
                        Math.min(overlayBottom, MARGIN.top + plotHeight) - Math.max(MARGIN.top, overlayTop)
                    );
                }

                // Draw intersection region (if both widths are set)
                if (xWidthPx > 0 && yWidthPx > 0) {
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)'; // White for intersection
                    ctx.fillRect(
                        Math.max(MARGIN.left, overlayLeft),
                        Math.max(MARGIN.top, overlayTop),
                        Math.min(overlayRight, MARGIN.left + plotWidth) - Math.max(MARGIN.left, overlayLeft),
                        Math.min(overlayBottom, MARGIN.top + plotHeight) - Math.max(MARGIN.top, overlayTop)
                    );
                }
            }

            // Draw crosshair lines (skip if exporting)
            if (!skipCrosshair) {
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

        // Get cursor widths
        const angleWidthPx = this.cursorWidth.angle || 0;
        const energyWidthEV = this.cursorWidth.energy || 0;

        // Convert energy width from eV to pixels
        let energyWidthPx = 0;
        if (energyWidthEV > 0 && this.axes.energy && this.axes.energy.length > 1) {
            const eMin = this.axes.energy[0];
            const eMax = this.axes.energy[this.axes.energy.length - 1];
            const eRange = Math.abs(eMax - eMin);
            const pxPerEV = height / eRange;
            energyWidthPx = Math.round(energyWidthEV * pxPerEV);
        }

        // Calculate integration ranges (clamp to valid indices)
        const xStart = Math.max(0, cursorX - angleWidthPx);
        const xEnd = Math.min(width - 1, cursorX + angleWidthPx);
        const yStart = Math.max(0, cursorY - energyWidthPx);
        const yEnd = Math.min(height - 1, cursorY + energyWidthPx);

        // Extract EDC (vertical cut: sum over angle range at each energy)
        const edc = new Float32Array(height);
        for (let i = 0; i < height; i++) {
            let sum = 0;
            for (let j = xStart; j <= xEnd; j++) {
                sum += this.data[i * width + j];
            }
            edc[i] = sum;
        }

        // Extract MDC (horizontal cut: sum over energy range at each angle)
        const mdc = new Float32Array(width);
        for (let j = 0; j < width; j++) {
            let sum = 0;
            for (let i = yStart; i <= yEnd; i++) {
                sum += this.data[i * width + j];
            }
            mdc[j] = sum;
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

        // Get cursor widths
        const angleWidthPx = this.cursorWidth.angle || 0;
        const scanWidthPx = this.cursorWidth.scan || 0;
        const energyWidthEV = this.cursorWidth.energy || 0;

        // Convert energy width from eV to pixels
        let energyWidthPx = 0;
        if (energyWidthEV > 0 && this.axes.energy && this.axes.energy.length > 1) {
            const eMin = this.axes.energy[0];
            const eMax = this.axes.energy[this.axes.energy.length - 1];
            const eRange = Math.abs(eMax - eMin);
            const pxPerEV = height / eRange;
            energyWidthPx = Math.round(energyWidthEV * pxPerEV);
        }

        // Calculate integration ranges (clamp to valid indices)
        const xStart = Math.max(0, idxX - angleWidthPx);
        const xEnd = Math.min(width - 1, idxX + angleWidthPx);
        const yStart = Math.max(0, idxY - energyWidthPx);
        const yEnd = Math.min(height - 1, idxY + energyWidthPx);
        const zStart = Math.max(0, idxZ - scanWidthPx);
        const zEnd = Math.min(depth - 1, idxZ + scanWidthPx);

        // Extract EDC from XY slice (sum over angle range at each energy, integrating over scan range too)
        const edc = new Float32Array(height);
        for (let i = 0; i < height; i++) {
            let sum = 0;
            for (let j = xStart; j <= xEnd; j++) {
                // For each scan in range, sum the values
                for (let k = zStart; k <= zEnd; k++) {
                    // data[i, j, k] in flattened format: i * (width * depth) + j * depth + k
                    const idx = i * (width * depth) + j * depth + k;
                    sum += this.data[idx];
                }
            }
            edc[i] = sum;
        }

        // Extract MDC from XY slice (sum over energy range at each angle, integrating over scan range too)
        const mdc = new Float32Array(width);
        for (let j = 0; j < width; j++) {
            let sum = 0;
            for (let i = yStart; i <= yEnd; i++) {
                for (let k = zStart; k <= zEnd; k++) {
                    const idx = i * (width * depth) + j * depth + k;
                    sum += this.data[idx];
                }
            }
            mdc[j] = sum;
        }

        // Extract Scan profile (sum over energy and angle ranges at each scan position)
        const scanProfile = new Float32Array(depth);
        for (let k = 0; k < depth; k++) {
            let sum = 0;
            for (let i = yStart; i <= yEnd; i++) {
                for (let j = xStart; j <= xEnd; j++) {
                    const idx = i * (width * depth) + j * depth + k;
                    sum += this.data[idx];
                }
            }
            scanProfile[k] = sum;
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
        ctx.fillStyle = this.getThemeColor('--bg-plot-panel', '#0a0a0a');
        ctx.fillRect(padding.left, padding.top, plotW, plotH);

        // Draw cursor indicator line (before profile so profile is on top)
        if (cursorIndex !== null && cursorIndex >= 0 && cursorIndex < data.length) {
            const cursorX = padding.left + (cursorIndex / (data.length - 1)) * plotW;
            ctx.strokeStyle = this.getThemeColor('--text-primary', '#fff'); // Use primary text color for cursor line roughly
            // Or better: use a contrasting color or specific var. 
            // In dark mode white is good. In light mode black is good. --text-primary fits.
            // But let's set alpha if needed. 
            // Actually, let's keep it simple:
            const cursorBaseWithAlpha = this.getThemeColor('--text-primary', '#e0e0e0');
            ctx.strokeStyle = cursorBaseWithAlpha;
            // Note: Canvas strokeStyle string parsing handles vars if they are color strings.
            // But --text-primary is a hex. We might want alpha. 
            // Let's just use the solid color for now or rely on canvas globalAlpha if we wanted translucency.
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

            const axisColor = this.getThemeColor('--text-secondary', '#888');
            ctx.strokeStyle = axisColor;
            ctx.fillStyle = axisColor;
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
        ctx.fillStyle = this.getThemeColor('--text-secondary', '#aaa');
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

        // Update state
        this.enhancement.contrastMin = this.contrastMin;
        this.enhancement.contrastMax = this.contrastMax;

        // Notify callback
        if (this.onEnhancementChange) {
            this.onEnhancementChange({
                contrastMin: this.contrastMin,
                contrastMax: this.contrastMax
            });
        }

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

    /**
     * Extract a slice averaged over a range of indices along the fixed dimension.
     * Uses mean instead of sum to preserve colormap contrast.
     * @param {number} dim - Dimension to fix (0=energy, 1=angle, 2=scan)
     * @param {number} centerIndex - Center index for the slice
     * @param {number} halfWidth - Half-width of integration range (in pixels)
     * @returns {Float32Array} - Averaged slice
     */
    extractSliceIntegrated(dim, centerIndex, halfWidth) {
        if (!this.data || this.ndim !== 3) return null;
        if (halfWidth <= 0) return this.extractSlice(dim, centerIndex);

        const [d0, d1, d2] = this.shape;
        const maxIdx = [d0, d1, d2][dim] - 1;
        const startIdx = Math.max(0, centerIndex - halfWidth);
        const endIdx = Math.min(maxIdx, centerIndex + halfWidth);
        const count = endIdx - startIdx + 1;

        if (dim === 0) {
            // Fix dim 0 (Energy). Slice is (d1, d2). Average over energy range.
            const size = d1 * d2;
            const slice = new Float32Array(size);
            for (let idx = startIdx; idx <= endIdx; idx++) {
                const start = idx * size;
                for (let i = 0; i < size; i++) {
                    slice[i] += this.data[start + i];
                }
            }
            // Divide by count to get mean
            for (let i = 0; i < size; i++) {
                slice[i] /= count;
            }
            return slice;
        } else if (dim === 1) {
            // Fix dim 1 (Angle). Slice is (d0, d2). Average over angle range.
            const size = d0 * d2;
            const slice = new Float32Array(size);
            for (let idx = startIdx; idx <= endIdx; idx++) {
                let ptr = 0;
                for (let i0 = 0; i0 < d0; i0++) {
                    const rowStart = i0 * (d1 * d2) + idx * d2;
                    for (let i2 = 0; i2 < d2; i2++) {
                        slice[ptr++] += this.data[rowStart + i2];
                    }
                }
            }
            for (let i = 0; i < size; i++) {
                slice[i] /= count;
            }
            return slice;
        } else if (dim === 2) {
            // Fix dim 2 (Scan). Slice is (d0, d1). Average over scan range.
            const size = d0 * d1;
            const slice = new Float32Array(size);
            for (let idx = startIdx; idx <= endIdx; idx++) {
                let ptr = 0;
                for (let i0 = 0; i0 < d0; i0++) {
                    for (let i1 = 0; i1 < d1; i1++) {
                        const dataIdx = i0 * (d1 * d2) + i1 * d2 + idx;
                        slice[ptr++] += this.data[dataIdx];
                    }
                }
            }
            for (let i = 0; i < size; i++) {
                slice[i] /= count;
            }
            return slice;
        }
        return null;
    }

    /**
     * Export figures without crosshair, as individual files for 3D data.
     * @param {string} format - 'png' or 'svg'
     */
    exportFigures(format = 'png') {
        if (!this.data) return;

        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');

        if (this.ndim === 2) {
            // 2D: Render to export canvas without crosshair
            const { canvas: exportCanvas, viewParams } = this._renderViewForExport(0, '2d');
            this._downloadCanvas(exportCanvas, `arpes_2d_${timestamp}`, format, viewParams);
        } else if (this.ndim === 3) {
            // 3D: Export each view as individual file
            const viewNames = ['xy', 'xz', 'yz'];
            for (let i = 0; i < 3; i++) {
                const { canvas: exportCanvas, viewParams } = this._renderViewForExport(i, viewNames[i]);
                this._downloadCanvas(exportCanvas, `arpes_${viewNames[i]}_${timestamp}`, format, viewParams);
            }
        }

        // Redraw with crosshair after export
        this.draw();
    }

    /**
     * Render a view to an export canvas without crosshair.
     * @param {number} viewIndex - Index of the view (0, 1, 2 for 3D)
     * @param {string} viewType - 'xy', 'xz', 'yz', or '2d'
     * @returns {HTMLCanvasElement} Canvas ready for export
     */
    _renderViewForExport(viewIndex, viewType) {
        const sourceCanvas = this.canvases[viewIndex];
        const sourceRect = sourceCanvas.parentElement.getBoundingClientRect();

        // Publication quality: 300 DPI
        // Standard screen is ~96 DPI, so we need ~3.125x scaling
        // Using 3x for clean integer scaling, resulting in ~288 DPI at typical screen resolution
        const EXPORT_SCALE = 3;

        // Create high-resolution export canvas
        const exportCanvas = document.createElement('canvas');
        const displayWidth = sourceRect.width;
        const displayHeight = sourceRect.height;
        exportCanvas.width = displayWidth * EXPORT_SCALE;
        exportCanvas.height = displayHeight * EXPORT_SCALE;

        // Get the data and parameters for this view
        let sliceData, dataWidth, dataHeight, xAxis, yAxis, xLabel, yLabel, crosshair, overlayWidths;

        if (this.ndim === 2) {
            sliceData = this.data;
            dataWidth = this.shape[1];
            dataHeight = this.shape[0];
            xAxis = this.axes.kx;
            yAxis = this.axes.energy;
            xLabel = this.axisLabels.x || "Angle";
            yLabel = this.axisLabels.y || "Energy (eV)";
            crosshair = [this.sliceIndices[1], this.sliceIndices[0]];
            overlayWidths = null;
        } else {
            const [nz, ny, nx] = this.shape;
            const idxY = this.sliceIndices[0];
            const idxX = this.sliceIndices[1];
            const idxZ = this.sliceIndices[2];

            const scanWidthPx = this.cursorWidth.scan || 0;
            const angleWidthPx = this.cursorWidth.angle || 0;
            const energyWidthEV = this.cursorWidth.energy || 0;
            let energyWidthPx = 0;
            if (energyWidthEV > 0 && this.axes.energy && this.axes.energy.length > 1) {
                const eMin = this.axes.energy[0];
                const eMax = this.axes.energy[this.axes.energy.length - 1];
                const eRange = Math.abs(eMax - eMin);
                const pxPerEV = this.shape[0] / eRange;
                energyWidthPx = Math.round(energyWidthEV * pxPerEV);
            }

            if (viewType === 'xy') {
                sliceData = this.extractSliceIntegrated(2, idxZ, scanWidthPx);
                dataWidth = this.shape[1];
                dataHeight = this.shape[0];
                xAxis = this.axes.kx;
                yAxis = this.axes.energy;
                xLabel = this.axisLabels.x || "Angle";
                yLabel = this.axisLabels.y || "Energy (eV)";
                crosshair = [idxX, idxY];
                overlayWidths = { x: angleWidthPx, y: energyWidthPx };
            } else if (viewType === 'xz') {
                sliceData = this.extractSliceIntegrated(1, idxX, angleWidthPx);
                dataWidth = this.shape[2];
                dataHeight = this.shape[0];
                xAxis = this.axes.ky;
                yAxis = this.axes.energy;
                xLabel = this.axisLabels.z || "Scan";
                yLabel = this.axisLabels.y || "Energy (eV)";
                crosshair = [idxZ, idxY];
                overlayWidths = { x: scanWidthPx, y: energyWidthPx };
            } else if (viewType === 'yz') {
                sliceData = this.extractSliceIntegrated(0, idxY, energyWidthPx);
                dataWidth = this.shape[2];
                dataHeight = this.shape[1];
                xAxis = this.axes.ky;
                yAxis = this.axes.kx;
                xLabel = this.axisLabels.z || "Scan";
                yLabel = this.axisLabels.x || "Angle";
                crosshair = [idxZ, idxX];
                overlayWidths = { x: scanWidthPx, y: angleWidthPx };
            }
        }

        // Draw to export canvas with skipCrosshair=true, passing explicit dimensions and scale
        const exportDimensions = { width: displayWidth, height: displayHeight };
        this.draw2D(exportCanvas, sliceData, dataWidth, dataHeight, xAxis, yAxis, xLabel, yLabel, crosshair, overlayWidths, true, exportDimensions, EXPORT_SCALE);

        // Return canvas and view parameters for SVG generation
        return {
            canvas: exportCanvas,
            viewParams: { xAxis, yAxis, xLabel, yLabel }
        };
    }

    /**
     * Download canvas as file (PNG or SVG).
     * For SVG: Creates layered SVG with separate groups for image and vector axes.
     * @param {HTMLCanvasElement} canvas - Canvas to export
     * @param {string} filename - Base filename (without extension)
     * @param {string} format - 'png' or 'svg'
     * @param {Object} viewParams - Parameters for generating vector axes (xAxis, yAxis, xLabel, yLabel)
     */
    _downloadCanvas(canvas, filename, format, viewParams = null) {
        const link = document.createElement('a');

        if (format === 'svg') {
            const svgContent = this._generateLayeredSVG(canvas, viewParams);
            const blob = new Blob([svgContent], { type: 'image/svg+xml' });
            link.href = URL.createObjectURL(blob);
            link.download = `${filename}.svg`;
        } else {
            // PNG export
            link.href = canvas.toDataURL('image/png');
            link.download = `${filename}.png`;
        }

        link.click();

        // Clean up object URL if created
        if (format === 'svg') {
            setTimeout(() => URL.revokeObjectURL(link.href), 100);
        }
    }

    /**
     * Generate layered SVG with separate groups for image and vector elements.
     * Layers: Heatmap (raster), Axes (vector), Labels (vector text)
     * @param {HTMLCanvasElement} canvas - Source canvas with heatmap
     * @param {Object} viewParams - Axis parameters for vector elements
     * @returns {string} SVG content string
     */
    _generateLayeredSVG(canvas, viewParams) {
        const width = canvas.width;
        const height = canvas.height;
        const scale = 3; // Match export scale

        // Calculate plot area in export coordinates
        const margin = {
            top: MARGIN.top * scale,
            right: MARGIN.right * scale,
            bottom: MARGIN.bottom * scale,
            left: MARGIN.left * scale
        };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;

        // Create heatmap-only canvas (without axes)
        const heatmapCanvas = document.createElement('canvas');
        heatmapCanvas.width = plotWidth;
        heatmapCanvas.height = plotHeight;
        const heatmapCtx = heatmapCanvas.getContext('2d');

        // Copy only the plot area from the full canvas
        heatmapCtx.drawImage(canvas,
            margin.left, margin.top, plotWidth, plotHeight,
            0, 0, plotWidth, plotHeight);
        const heatmapDataUrl = heatmapCanvas.toDataURL('image/png');

        // Build SVG with layers
        let svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
     width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  
  <!-- Heatmap Layer (Raster) -->
  <g id="heatmap-layer">
    <image x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" 
           xlink:href="${heatmapDataUrl}" preserveAspectRatio="none"/>
  </g>
  
  <!-- Axes Layer (Vector) -->
  <g id="axes-layer" stroke="#888888" stroke-width="${scale}" fill="none">
    <!-- X Axis -->
    <line x1="${margin.left}" y1="${margin.top + plotHeight}" 
          x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}"/>
    <!-- Y Axis -->
    <line x1="${margin.left}" y1="${margin.top}" 
          x2="${margin.left}" y2="${margin.top + plotHeight}"/>
  </g>
`;

        // Add tick marks and labels if viewParams provided
        if (viewParams && viewParams.xAxis && viewParams.yAxis) {
            const fontSize = 12 * scale;
            const labelFontSize = 14 * scale;
            const tickSize = 5 * scale;

            // X-axis ticks and labels
            const xMin = viewParams.xAxis[0];
            const xMax = viewParams.xAxis[viewParams.xAxis.length - 1];
            const xTicks = this.getNiceTicks(xMin, xMax, Math.floor(plotWidth / (60 * scale)));

            svg += `
  <!-- X-Axis Ticks Layer (Vector) -->
  <g id="x-ticks-layer" stroke="#888888" stroke-width="${scale}" fill="none">
`;
            xTicks.forEach(val => {
                const t = (val - xMin) / (xMax - xMin);
                if (t >= 0 && t <= 1) {
                    const x = margin.left + t * plotWidth;
                    svg += `    <line x1="${x}" y1="${margin.top + plotHeight}" x2="${x}" y2="${margin.top + plotHeight + tickSize}"/>
`;
                }
            });
            svg += `  </g>
`;

            // X-axis tick labels
            svg += `
  <!-- X-Axis Labels Layer (Vector Text) -->
  <g id="x-labels-layer" font-family="sans-serif" font-size="${fontSize}" fill="#CCCCCC" text-anchor="middle">
`;
            xTicks.forEach(val => {
                const t = (val - xMin) / (xMax - xMin);
                if (t >= 0 && t <= 1) {
                    const x = margin.left + t * plotWidth;
                    const label = Math.abs(val) < 1e-10 ? "0" :
                        Math.abs(val) < 0.01 ? val.toExponential(1) :
                            parseFloat(val.toFixed(2)).toString();
                    svg += `    <text x="${x}" y="${margin.top + plotHeight + tickSize + fontSize}">${label}</text>
`;
                }
            });
            svg += `  </g>
`;

            // X-axis label
            if (viewParams.xLabel) {
                svg += `
  <!-- X-Axis Title Layer (Vector Text) -->
  <g id="x-title-layer" font-family="sans-serif" font-size="${labelFontSize}" fill="#E0E0E0" text-anchor="middle">
    <text x="${margin.left + plotWidth / 2}" y="${margin.top + plotHeight + tickSize + fontSize * 2.2}">${viewParams.xLabel}</text>
  </g>
`;
            }

            // Y-axis ticks and labels
            const yMin = viewParams.yAxis[0];
            const yMax = viewParams.yAxis[viewParams.yAxis.length - 1];
            const yTicks = this.getNiceTicks(yMin, yMax, Math.floor(plotHeight / (40 * scale)));

            svg += `
  <!-- Y-Axis Ticks Layer (Vector) -->
  <g id="y-ticks-layer" stroke="#888888" stroke-width="${scale}" fill="none">
`;
            yTicks.forEach(val => {
                const t = (val - yMin) / (yMax - yMin);
                if (t >= 0 && t <= 1) {
                    const y = margin.top + plotHeight - t * plotHeight;
                    svg += `    <line x1="${margin.left}" y1="${y}" x2="${margin.left - tickSize}" y2="${y}"/>
`;
                }
            });
            svg += `  </g>
`;

            // Y-axis tick labels
            svg += `
  <!-- Y-Axis Labels Layer (Vector Text) -->
  <g id="y-labels-layer" font-family="sans-serif" font-size="${fontSize}" fill="#CCCCCC" text-anchor="end">
`;
            yTicks.forEach(val => {
                const t = (val - yMin) / (yMax - yMin);
                if (t >= 0 && t <= 1) {
                    const y = margin.top + plotHeight - t * plotHeight;
                    const label = Math.abs(val) < 1e-10 ? "0" :
                        Math.abs(val) < 0.01 ? val.toExponential(1) :
                            parseFloat(val.toFixed(2)).toString();
                    svg += `    <text x="${margin.left - tickSize - 3 * scale}" y="${y + fontSize / 3}">${label}</text>
`;
                }
            });
            svg += `  </g>
`;

            // Y-axis label (rotated)
            if (viewParams.yLabel) {
                const yLabelX = fontSize;
                const yLabelY = margin.top + plotHeight / 2;
                svg += `
  <!-- Y-Axis Title Layer (Vector Text) -->
  <g id="y-title-layer" font-family="sans-serif" font-size="${labelFontSize}" fill="#E0E0E0" text-anchor="middle">
    <text x="${yLabelX}" y="${yLabelY}" transform="rotate(-90, ${yLabelX}, ${yLabelY})">${viewParams.yLabel}</text>
  </g>
`;
            }
        }

        svg += `</svg>`;
        return svg;
    }

    // Keep old method for backward compatibility
    exportImage() {
        this.exportFigures('png');
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
