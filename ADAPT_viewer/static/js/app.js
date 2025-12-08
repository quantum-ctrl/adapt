import { DataLoader } from './data_loader.js';
import { Visualizer } from './visualizer.js';
import { KConverter } from './converter.js';
import * as Physics from './physics.js';

class App {
    constructor() {
        this.dataLoader = new DataLoader();
        this.visualizer = new Visualizer(document.getElementById('plot-container'));

        this.originalData = null;
        this.originalMeta = null;

        this.calibration = {
            angleOffsetKx: 0,
            angleOffsetKy: 0,
            fermiOffset: 0
        };

        this.lastCursor = { x: 0, y: 0, z: 0, val: 0 };
        this.selectedFilePath = null;

        this.ui = {
            fileBrowserContainer: document.getElementById('file-browser-container'),
            loadBtn: document.getElementById('load-btn'),
            colormapSelect: document.getElementById('colormap-select'),
            cminSlider: document.getElementById('cmin-slider'),
            cmaxSlider: document.getElementById('cmax-slider'),
            cminVal: document.getElementById('cmin-val'),
            cmaxVal: document.getElementById('cmax-val'),
            autoContrastBtn: document.getElementById('auto-contrast-btn'),
            resetContrastBtn: document.getElementById('reset-contrast-btn'),
            cursorCoords: document.getElementById('cursor-coords'),
            dataInfo: document.getElementById('data-info'),
            axisControls: document.getElementById('axis-controls'),
            calibrationControls: document.getElementById('calibration-controls'),

            xAxisType: document.getElementById('x-axis-type'),
            yAxisType: document.getElementById('y-axis-type'),
            zAxisType: document.getElementById('z-axis-type'),
            hvInput: document.getElementById('hv-input'),
            workFuncInput: document.getElementById('work-func-input'),
            innerPotInput: document.getElementById('inner-pot-input'),
            setAngleZeroBtn: document.getElementById('set-angle-zero-btn'),
            setFermiBtn: document.getElementById('set-fermi-btn'),

            // Crop controls
            cropBtn: document.getElementById('crop-btn'),
            confirmCropBtn: document.getElementById('confirm-crop-btn'),
            cancelCropBtn: document.getElementById('cancel-crop-btn'),
            cropActions: document.getElementById('crop-actions'),

            // Enhancement controls
            enhanceToggleHeader: document.getElementById('enhance-toggle-header'),
            enhanceControls: document.getElementById('enhance-controls'),
            enhanceHeaderIcon: document.getElementById('enhance-header-icon'),
            enhanceEnableCheck: document.getElementById('enhance-enable-check'),
            smoothSlider: document.getElementById('smooth-slider'),
            smoothVal: document.getElementById('smooth-val'),
            sharpenSlider: document.getElementById('sharpen-slider'),
            sharpenVal: document.getElementById('sharpen-val'),
            bgSlider: document.getElementById('bg-slider'),
            bgVal: document.getElementById('bg-val'),
            claheEnableCheck: document.getElementById('clahe-enable-check'),
            claheControls: document.getElementById('clahe-controls'),
            claheSlider: document.getElementById('clahe-slider'),
            claheVal: document.getElementById('clahe-val'),

            // Curvature controls
            curvatureEnableCheck: document.getElementById('curvature-enable-check'),
            curvatureControls: document.getElementById('curvature-controls'),
            curvatureSlider: document.getElementById('curvature-slider'),
            curvatureVal: document.getElementById('curvature-val'),

            // Parameter controls
            efInput: document.getElementById('ef-input'),
            thetaInput: document.getElementById('theta-input'),
            scanInput: document.getElementById('scan-input'),
            scanInputRow: document.getElementById('scan-input-row'),
            getFromCursorBtn: document.getElementById('get-from-cursor-btn'),

            // Alignment controls
            alignEnergyBtn: document.getElementById('align-energy-btn'),
            alignThetaBtn: document.getElementById('align-theta-btn')
        };

        this.init();
    }

    async init() {
        this.setupEventListeners();

        // Initialize File Browser in Modal (if available)
        // Note: FileBrowser may not be defined if the feature is not implemented
        const fbContainer = document.getElementById('file-browser-body');
        if (fbContainer && typeof FileBrowser !== 'undefined') {
            try {
                this.fileBrowser = new FileBrowser(fbContainer, (path) => {
                    this.handleFileSelection(path);
                });
            } catch (e) {
                console.warn("FileBrowser initialization failed:", e.message);
            }
        }

        this.ui.loadBtn.disabled = true; // Disable load button initially

        // Force enable enhancement controls to ensure they are not stuck
        this.setControlsDisabled(false);
        console.log("Enhancement controls initialized:", this.ui.smoothSlider);

        // CRITICAL FIX: Sync initial enhancement state from HTML to Visualizer
        // The checkbox is checked by default in HTML, but Visualizer starts with enabled:false
        // We need to call updateEnhancement once to sync the state
        setTimeout(() => {
            if (this.ui.enhanceEnableCheck && this.ui.enhanceEnableCheck.checked) {
                // Trigger the change event to sync state
                this.ui.enhanceEnableCheck.dispatchEvent(new Event('change'));
            }
        }, 100);

        // Check for session parameter from ADAPT Browser integration
        await this.checkForSession();
    }

    /**
     * Check for session parameter and auto-load file from ADAPT Browser.
     * 
     * When ADAPT Browser opens Viewer with ?session=1, this method reads
     * the session file and automatically loads the selected data file.
     */
    async checkForSession() {
        const urlParams = new URLSearchParams(window.location.search);
        const sessionParam = urlParams.get('session');

        if (sessionParam !== '1') {
            console.log("No session parameter, skipping auto-load");
            return;
        }

        console.log("Session parameter detected, loading from ADAPT Browser...");

        try {
            // Show loading state
            if (this.ui.dataInfo) {
                this.ui.dataInfo.textContent = "Loading from ADAPT Browser...";
            }

            // Fetch session data from backend
            const sessionData = await this.dataLoader.loadSession();

            if (!sessionData || !sessionData.file_path) {
                console.log("No valid session data found");
                if (this.ui.dataInfo) {
                    this.ui.dataInfo.textContent = "No session file found. Please select a file.";
                }
                return;
            }

            // Set the file path and trigger load
            const filePath = sessionData.file_path;
            console.log("Session file path:", filePath);

            // Store the path and load the file
            this.selectedFilePath = filePath;

            // Update UI with file info
            const basename = filePath.split(/[/\\]/).pop();
            document.getElementById('selected-file-label').textContent = basename;

            // Load the file
            await this.loadSelectedFile();

            // Log successful session load
            console.log("Successfully loaded file from session:", basename);

            // Clear session parameter from URL without reload (cleaner UX)
            const newUrl = window.location.pathname;
            window.history.replaceState({}, document.title, newUrl);

        } catch (error) {
            console.error("Session load error:", error);
            if (this.ui.dataInfo) {
                this.ui.dataInfo.innerHTML = `<span style="color: #ff5252;">Session Error: ${error.message}</span>`;
            }
        }
    }

    setupEventListeners() {
        // Window Resize
        window.addEventListener('resize', () => {
            if (this.visualizer) this.visualizer.onResize();
        });

        // Local File Input
        const localBtn = document.getElementById('local-file-btn');
        const fileInput = document.getElementById('system-file-input');

        localBtn.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.selectedFileObj = e.target.files[0];
                this.selectedFilePath = null; // Clear server path

                document.getElementById('selected-file-label').textContent = this.selectedFileObj.name;
                // Auto-load
                this.loadSelectedFile();
            }
        });

        this.ui.loadBtn.addEventListener('click', () => this.loadSelectedFile());

        this.ui.colormapSelect.addEventListener('change', (e) => {
            this.visualizer.setColormap(e.target.value);
        });

        document.getElementById('invert-colormap-check').addEventListener('change', (e) => {
            this.visualizer.setInverted(e.target.checked);
        });

        this.ui.cminSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            const maxVal = parseInt(this.ui.cmaxSlider.value);
            // Prevent min slider from exceeding max slider
            if (val > maxVal) {
                e.target.value = maxVal;
                return;
            }
            this.ui.cminVal.textContent = val;
            this.visualizer.setContrastMin(val);
        });

        this.ui.cmaxSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            const minVal = parseInt(this.ui.cminSlider.value);
            // Prevent max slider from going below min slider
            if (val < minVal) {
                e.target.value = minVal;
                return;
            }
            this.ui.cmaxVal.textContent = val;
            this.visualizer.setContrastMax(val);
        });

        this.ui.autoContrastBtn.addEventListener('click', () => {
            this.visualizer.autoContrast();
            this.updateContrastUI();
        });

        this.ui.resetContrastBtn.addEventListener('click', () => {
            this.visualizer.resetContrast();
            this.updateContrastUI();
        });

        // Axis & Calibration Controls
        ['xAxisType', 'yAxisType', 'zAxisType', 'hvInput', 'workFuncInput', 'innerPotInput'].forEach(key => {
            if (this.ui[key]) {
                this.ui[key].addEventListener('change', () => {
                    this.updateAxisLabelsFromType();
                    this.recalculateData();
                });
            }
        });

        // Custom Axis Labels
        ['x-axis-label', 'y-axis-label', 'z-axis-label'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('input', () => {
                    this.visualizer.setAxisLabels({
                        x: document.getElementById('x-axis-label').value,
                        y: document.getElementById('y-axis-label').value,
                        z: document.getElementById('z-axis-label').value
                    });
                });
            }
        });

        this.ui.setAngleZeroBtn.addEventListener('click', () => this.setAngleZero());
        this.ui.setFermiBtn.addEventListener('click', () => this.setFermiLevel());

        document.getElementById('angle-to-k-btn').addEventListener('click', () => {
            // Set X-axis to k
            this.ui.xAxisType.value = 'k';
            // Trigger recalculation
            this.recalculateData();
        });
        // Convert / k-space group visibility handling
        const convertGroup = document.getElementById('convert-controls');
        if (convertGroup) {
            // Show convert controls when axis UI is available and data loaded
            // We'll toggle visibility when data is loaded (in loadSelectedFile)
        }

        // Auto Set EF (Fermi-Dirac fit) button
        const autoSetEFBtn = document.getElementById('auto-set-ef-btn');
        if (autoSetEFBtn) {
            autoSetEFBtn.addEventListener('click', async () => {
                if (!this.originalData || !this.originalMeta) {
                    alert("No data loaded");
                    return;
                }
                if (this.originalMeta.data_info.ndim !== 2) {
                    alert("Auto Set EF currently supports 2D datasets only");
                    return;
                }

                // Prepare args for fit
                const height = this.originalMeta.data_info.shape[0];
                const width = this.originalMeta.data_info.shape[1];
                const energyAxis = this.originalMeta.axes.energy;

                try {
                    const fitRes = Physics.fitFermiEdge2D(this.originalData, width, height, energyAxis, { smoothing: parseFloat(this.ui.smoothSlider.value) || 1 });
                    if (fitRes && fitRes.status === 'success') {
                        // Set calibration offset so EF -> 0
                        this.calibration.fermiOffset = fitRes.ef;
                        // Recalculate to apply
                        this.recalculateData();
                        alert(`Auto Set EF: detected EF = ${fitRes.ef.toFixed(4)} eV`);
                    } else {
                        alert("Auto Set EF failed or fallback used");
                    }
                } catch (e) {
                    console.error("Auto Set EF error:", e);
                    alert("Auto Set EF error: " + e.message);
                }
            });
        }

        document.getElementById('reset-data-btn').addEventListener('click', () => {
            this.resetData();
        });

        document.getElementById('reset-calibration-btn').addEventListener('click', () => {
            this.resetData();
        });

        document.getElementById('export-btn').addEventListener('click', () => {
            this.visualizer.exportImage();
        });

        document.getElementById('export-profiles-btn').addEventListener('click', () => {
            this.exportProfiles();
        });

        const mappingSelect = document.getElementById('mapping-mode-select');
        const kzGroup = document.getElementById('kz-mode-group');
        const kzSelect = document.getElementById('kz-mode-select');

        mappingSelect.addEventListener('change', (e) => {
            kzGroup.style.display = e.target.value === 'kx-kz' ? 'block' : 'none';
            // If already in kx mode, trigger recalc?
            if (this.ui.xAxisType.value === 'kx') this.recalculateData();
        });

        kzSelect.addEventListener('change', () => {
            if (this.ui.xAxisType.value === 'kx') this.recalculateData();
        });

        // Listen for cursor updates from visualizer
        document.addEventListener('cursor-update', (e) => {
            const { x, y, z, val } = e.detail;

            // Store raw indices for calibration logic if needed, 
            // but we need physical values for calibration offsets.
            // Let's calculate physical values here.

            const axes = this.visualizer.axes;
            let xVal = x, yVal = y, zVal = z;

            if (axes) {
                // 2D
                if (!z && axes.kx && axes.energy) {
                    // x is index in kx (Angle)
                    // y is index in energy
                    if (x >= 0 && x < axes.kx.length) xVal = axes.kx[x];
                    if (y >= 0 && y < axes.energy.length) yVal = axes.energy[y];
                }
                // 3D
                else if (z !== undefined && axes.kx && axes.energy && axes.ky) {
                    // x is Angle (dim1)
                    // y is Energy (dim0)
                    // z is Scan (dim2)
                    if (x >= 0 && x < axes.kx.length) xVal = axes.kx[x];
                    if (y >= 0 && y < axes.energy.length) yVal = axes.energy[y];
                    if (z >= 0 && z < axes.ky.length) zVal = axes.ky[z];
                }
            }

            this.lastCursor = { x: xVal, y: yVal, z: zVal, val };

            if (this.ui.cursorCoords) {
                let text = `X: ${xVal.toFixed(3)} | Y: ${yVal.toFixed(3)}`;
                if (z !== undefined) text += ` | Z: ${zVal.toFixed(3)}`;
                text += ` | Val: ${val.toExponential(2)}`;
                this.ui.cursorCoords.textContent = text;
            }

            // Automatically update EF, theta, and scan textboxes
            if (this.ui.efInput) {
                this.ui.efInput.value = yVal.toFixed(4);
            }
            if (this.ui.thetaInput) {
                this.ui.thetaInput.value = xVal.toFixed(2);
            }
            if (this.ui.scanInput && z !== undefined) {
                this.ui.scanInput.value = zVal.toFixed(2);
            }
        });

        // Crop mode event listeners
        if (this.ui.cropBtn) {
            this.ui.cropBtn.addEventListener('click', () => this.startCrop());
        }
        if (this.ui.confirmCropBtn) {
            this.ui.confirmCropBtn.addEventListener('click', () => this.confirmCrop());
        }
        if (this.ui.cancelCropBtn) {
            this.ui.cancelCropBtn.addEventListener('click', () => this.cancelCrop());
        }

        document.getElementById('cancel-calibration-btn').addEventListener('click', () => this.cancelCalibration());
        const confirmCalBtn = document.getElementById('confirm-calibration-btn');
        if (confirmCalBtn) {
            confirmCalBtn.addEventListener('click', () => this.confirmCalibration());
        }

        // Listen for crop mode events from visualizer
        document.addEventListener('crop-mode-entered', () => {
            this.onCropModeEntered();
        });
        document.addEventListener('crop-mode-exited', () => {
            this.onCropModeExited();
        });

        // Enhancement Controls
        if (this.ui.enhanceToggleHeader) {
            this.ui.enhanceToggleHeader.addEventListener('click', () => {
                const isHidden = this.ui.enhanceControls.style.display === 'none';
                this.ui.enhanceControls.style.display = isHidden ? 'block' : 'none';
                this.ui.enhanceHeaderIcon.classList.toggle('rotate-icon', isHidden);
            });
        }

        const updateEnhancement = () => {
            const opts = {
                enabled: this.ui.enhanceEnableCheck.checked,
                smoothing: parseFloat(this.ui.smoothSlider.value),
                sharpen: parseFloat(this.ui.sharpenSlider.value),
                background: parseInt(this.ui.bgSlider.value),
                clahe: this.ui.claheEnableCheck.checked,
                claheClip: parseFloat(this.ui.claheSlider.value),
                curvature: this.ui.curvatureEnableCheck ? this.ui.curvatureEnableCheck.checked : false,
                curvatureStrength: this.ui.curvatureSlider ? parseFloat(this.ui.curvatureSlider.value) : 1.0
            };
            console.log("App.updateEnhancement:", opts);
            this.visualizer.setEnhancement(opts);
        };

        this.ui.enhanceEnableCheck.addEventListener('change', updateEnhancement);

        // Note: Auto-align checkbox and re-detect button removed; use "Auto Set EF" in Convert controls.

        this.ui.smoothSlider.addEventListener('input', (e) => {
            console.log("Smooth slider input:", e.target.value);
            if (this.ui.smoothVal) this.ui.smoothVal.textContent = e.target.value;
            if (!this.ui.enhanceEnableCheck.checked) {
                this.ui.enhanceEnableCheck.checked = true;
            }
            updateEnhancement();
        });

        this.ui.sharpenSlider.addEventListener('input', (e) => {
            if (this.ui.sharpenVal) this.ui.sharpenVal.textContent = e.target.value;
            if (!this.ui.enhanceEnableCheck.checked) {
                this.ui.enhanceEnableCheck.checked = true;
            }
            updateEnhancement();
        });

        this.ui.bgSlider.addEventListener('input', (e) => {
            if (this.ui.bgVal) this.ui.bgVal.textContent = e.target.value;
            if (!this.ui.enhanceEnableCheck.checked) {
                this.ui.enhanceEnableCheck.checked = true;
            }
            updateEnhancement();
        });

        this.ui.claheEnableCheck.addEventListener('change', (e) => {
            this.ui.claheControls.style.display = e.target.checked ? 'block' : 'none';
            updateEnhancement();
        });

        this.ui.claheSlider.addEventListener('input', (e) => {
            this.ui.claheVal.textContent = e.target.value;
            updateEnhancement();
        });

        // Curvature controls
        if (this.ui.curvatureEnableCheck) {
            this.ui.curvatureEnableCheck.addEventListener('change', (e) => {
                if (this.ui.curvatureControls) {
                    this.ui.curvatureControls.style.display = e.target.checked ? 'block' : 'none';
                }
                if (!this.ui.enhanceEnableCheck.checked) {
                    this.ui.enhanceEnableCheck.checked = true;
                }
                updateEnhancement();
            });
        }

        if (this.ui.curvatureSlider) {
            this.ui.curvatureSlider.addEventListener('input', (e) => {
                if (this.ui.curvatureVal) this.ui.curvatureVal.textContent = e.target.value;
                updateEnhancement();
            });
        }

        // Get EF, theta, and scan from Cursor button
        if (this.ui.getFromCursorBtn) {
            this.ui.getFromCursorBtn.addEventListener('click', () => {
                // lastCursor.y is energy, lastCursor.x is theta (angle), lastCursor.z is scan
                if (this.ui.efInput) {
                    this.ui.efInput.value = this.lastCursor.y.toFixed(4);
                }
                if (this.ui.thetaInput) {
                    this.ui.thetaInput.value = this.lastCursor.x.toFixed(2);
                }
                // For 3D data, also populate scan value
                if (this.ui.scanInput && this.lastCursor.z !== undefined) {
                    this.ui.scanInput.value = this.lastCursor.z.toFixed(2);
                }
            });
        }

        // Align Energy button - align energy axis to EF value
        if (this.ui.alignEnergyBtn) {
            this.ui.alignEnergyBtn.addEventListener('click', () => {
                const ef = parseFloat(this.ui.efInput.value) || 0;
                this.calibration.fermiOffset = ef;
                this.recalculateData();
            });
        }

        // Align Theta button - align theta axis (and scan axis for 3D) to theta/scan values
        if (this.ui.alignThetaBtn) {
            this.ui.alignThetaBtn.addEventListener('click', () => {
                const theta = parseFloat(this.ui.thetaInput.value) || 0;
                this.calibration.angleOffsetKx = theta;
                // For 3D data, also align scan axis
                if (this.originalMeta && this.originalMeta.data_info.ndim === 3) {
                    const scan = parseFloat(this.ui.scanInput.value) || 0;
                    this.calibration.angleOffsetKy = scan;
                }
                this.recalculateData();
            });
        }
    }

    updateContrastUI() {
        const { min, max } = this.visualizer.getContrastRange();
        this.ui.cminSlider.value = min;
        this.ui.cmaxSlider.value = max;
        this.ui.cminVal.textContent = min;
        this.ui.cmaxVal.textContent = max;
    }

    async loadSelectedFile() {
        let filename = this.selectedFilePath;
        if (!filename && !this.selectedFileObj) return;

        this.ui.loadBtn.disabled = true;
        this.ui.loadBtn.textContent = "Loading...";
        if (this.ui.dataInfo) this.ui.dataInfo.textContent = `Loading...`;

        try {
            // Upload if local file
            if (this.selectedFileObj) {
                if (this.ui.dataInfo) this.ui.dataInfo.textContent = `Opening ${this.selectedFileObj.name}...`;
                filename = await this.uploadFile(this.selectedFileObj);
                this.selectedFilePath = filename; // Update path to server path
                this.selectedFileObj = null; // Clear obj so we don't re-upload unless changed
            }

            if (this.ui.dataInfo) this.ui.dataInfo.textContent = `Loading ${filename}...`;

            // Reset calibration
            this.calibration = { angleOffsetKx: 0, angleOffsetKy: 0, fermiOffset: 0 };

            // 1. Load Metadata
            const meta = await this.dataLoader.loadMetadata(filename);
            this.originalMeta = meta;

            // 2. Load Binary Data
            const data = await this.dataLoader.loadData(filename);
            this.originalData = data;

            // 3. Initial Recalculate (handles display)
            this.recalculateData();

            // Auto-adjust contrast after loading new data
            try {
                this.visualizer.autoContrast();
                this.updateContrastUI();
            } catch (e) {
                console.warn("Auto-contrast failed:", e);
            }

            // 4. Update UI
            // Show only basename in the top label
            const basename = filename.split(/[/\\]/).pop();
            if (this.ui.dataInfo) this.ui.dataInfo.textContent = `File: ${basename}`;

            this.ui.axisControls.style.display = 'block';
            this.ui.calibrationControls.style.display = 'block';

            // Populate Metadata Viewer with full info
            const metaSection = document.getElementById('metadata-section');
            if (metaSection) {
                metaSection.style.display = 'block';
                this.displayMetadata(meta, filename);
            }

            const is3D = meta.data_info.ndim === 3;
            document.getElementById('z-axis-group').style.display = is3D ? 'block' : 'none';
            document.getElementById('3d-mapping-controls').style.display = is3D ? 'block' : 'none';
            // Show/hide scan input row for 3D data
            if (this.ui.scanInputRow) {
                this.ui.scanInputRow.style.display = is3D ? 'flex' : 'none';
            }

            // Default axis assignment: for 3D data, set X to Angle, Y to Energy, Z to Scan.
            // User can change these via the Axis Management panel; reflect defaults in the UI.
            if (is3D) {
                if (this.ui.xAxisType) this.ui.xAxisType.value = 'angle';
                if (this.ui.yAxisType) this.ui.yAxisType.value = 'energy';
                if (this.ui.zAxisType) this.ui.zAxisType.value = 'scan';
            } else {
                // 2D defaults
                if (this.ui.xAxisType) this.ui.xAxisType.value = 'angle';
                if (this.ui.yAxisType) this.ui.yAxisType.value = 'energy';
            }

            // Reset mapping UI
            this.ui.xAxisType.value = 'angle';        // Reset Mapping UI
            this.ui.yAxisType.value = 'energy';
            document.getElementById('mapping-mode-select').value = 'kx-ky';
            document.getElementById('kz-mode-select').value = 'convert';
            document.getElementById('kz-mode-group').style.display = 'none';

            this.updateAxisLabelsFromType();

            // Auto-populate hv from metadata if available
            if (meta.metadata && meta.metadata.hv !== undefined && meta.metadata.hv !== null) {
                const hvVal = typeof meta.metadata.hv === 'number' ? meta.metadata.hv :
                    (Array.isArray(meta.metadata.hv) ? meta.metadata.hv[0] : parseFloat(meta.metadata.hv));
                if (!isNaN(hvVal) && this.ui.hvInput) {
                    this.ui.hvInput.value = hvVal.toFixed(1);
                }
            }

            // Recalculate
            this.recalculateData();

        } catch (error) {
            console.error("Error loading file:", error);
            if (this.ui.dataInfo) this.ui.dataInfo.innerHTML = `<span style="color: #ff5252;">Error: ${error.message}</span>`;
            alert(`Failed to load file: ${error.message}`);
        } finally {
            this.ui.loadBtn.disabled = false;
            this.ui.loadBtn.textContent = "Load Data";
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error("Upload failed");
        }

        const data = await response.json();
        return data.filename;
    }

    recalculateData() {
        if (!this.originalData || !this.originalMeta) return;

        const xAxisType = this.ui.xAxisType.value;
        const yAxisType = this.ui.yAxisType.value;
        const workFunc = parseFloat(this.ui.workFuncInput.value);
        const innerPot = parseFloat(this.ui.innerPotInput.value);
        const hv = parseFloat(this.ui.hvInput.value);

        let displayData = this.originalData;
        let displayMeta = JSON.parse(JSON.stringify(this.originalMeta)); // Deep copy

        // Apply Calibration Offsets to Original Axes
        // Note: 'kx' in original axes is usually Angle.
        // 'energy' is Energy.
        // Apply Calibration Offsets to Original Axes
        // Note: 'kx' in original axes is usually Angle.
        // 'energy' is Energy.
        if (displayMeta.axes.kx) {
            displayMeta.axes.kx = displayMeta.axes.kx.map(v => v - this.calibration.angleOffsetKx);
        }
        if (displayMeta.axes.ky) {
            // Only apply if ky is also an angle axis (heuristic: check label or just apply if user calibrated it)
            // For now, we apply if offset is non-zero, assuming user knew what they were doing.
            displayMeta.axes.ky = displayMeta.axes.ky.map(v => v - this.calibration.angleOffsetKy);
        }
        if (displayMeta.axes.energy) {
            displayMeta.axes.energy = displayMeta.axes.energy.map(v => v - this.calibration.fermiOffset);
        }

        // Check for Momentum Conversion (2D case for now)
        // If X-axis is 'kx' and original was 'angle'
        // Original axes: kx (Angle), energy (Energy)
        // Note: 'kx' in original axes usually means Angle.
        // If user selects 'kx', we want converted kx.

        if (displayMeta.data_info.ndim === 2) {
            if (xAxisType === 'k') {
                // Check if already in k-space (e.g. if we cropped a k-space image)
                const isAlreadyK = displayMeta.kx_unit === "1/Å";

                if (!isAlreadyK) {
                    const result = KConverter.convert2D(this.originalData, displayMeta, hv, workFunc);

                    displayData = result.data;
                    displayMeta.axes.kx = result.axes.kx;
                    // Energy axis is preserved/copied in convert2D but we already have it in displayMeta
                    displayMeta.data_info.shape = result.shape;
                    displayMeta.kx_unit = "1/Å";
                }
            }
        } else if (displayMeta.data_info.ndim === 3) {
            if (xAxisType === 'k') {
                // Check if already in k-space
                const isAlreadyK = displayMeta.kx_unit === "1/Å";

                if (!isAlreadyK) {
                    const mappingMode = document.getElementById('mapping-mode-select').value;
                    const kzMode = document.getElementById('kz-mode-select').value;

                    const result = KConverter.convert3D(
                        this.originalData, displayMeta,
                        hv, workFunc, innerPot,
                        mappingMode, kzMode
                    );

                    displayData = result.data;
                    displayMeta.axes.kx = result.axes.kx;
                    displayMeta.axes.ky = result.axes.ky;
                    displayMeta.data_info.shape = result.shape;
                    displayMeta.kx_unit = result.units.kx;
                    displayMeta.ky_unit = result.units.ky;
                }
            }
        }

        // Note: automatic in-place EF shifting removed; use the Convert -> Auto Set EF button which sets calibration.fermiOffset

        this.visualizer.setData(displayData, displayMeta);
    }

    setAngleZero() {
        if (this.ui.xAxisType.value !== 'angle') {
            alert("Please switch X-axis to 'Angle' to calibrate Angle Zero.");
            return;
        }

        this.enterCalibrationUI('angle');
    }

    setFermiLevel() {
        this.enterCalibrationUI('fermi');
    }

    enterCalibrationUI(type) {
        const success = this.visualizer.enterCalibrationMode(type);
        if (!success) {
            alert("Failed to enter calibration mode");
            return;
        }

        // Show actions, hide triggers
        document.getElementById('calibration-actions').style.display = 'flex';
        document.getElementById('set-angle-zero-btn').parentElement.style.display = 'none';

        // Disable other controls
        this.setControlsDisabled(true);
    }

    confirmCalibration() {
        const mode = this.visualizer.calibrationMode;
        const val = this.visualizer.getCalibrationValue();

        if (mode === 'angle') {
            // Check specific axis calibrations from visualizer
            // visualizer.calibrationPositions has keys 'kx', 'ky', 'energy'
            const positions = this.visualizer.calibrationPositions;

            if (positions.kx !== undefined) {
                this.calibration.angleOffsetKx = positions.kx;
            }

            if (positions.ky !== undefined) {
                this.calibration.angleOffsetKy = positions.ky;
            }

            // Fallback if no specific axis was touched but confirm was clicked (e.g. just center)
            // But usually dragging updates positions. 
            // If we just clicked "Set Angle Zero" and then "Confirm" without moving, 
            // we might want to set zero to current center? 
            // The visualizer initializes calibrationPositions to center.
            // So the above checks should cover it.

        } else if (mode === 'fermi') {
            // val is the absolute energy value where we want EF (zero) to be
            // So the offset is simply val (to shift that position to zero)
            // Check specific energy calibration
            const positions = this.visualizer.calibrationPositions;
            if (positions.energy !== undefined) {
                this.calibration.fermiOffset = positions.energy;
            } else {
                // Fallback to generic val if needed
                this.calibration.fermiOffset = val;
            }
        }

        this.exitCalibrationUI();
        this.recalculateData();
    }

    cancelCalibration() {
        this.exitCalibrationUI();
    }

    exitCalibrationUI() {
        this.visualizer.exitCalibrationMode();

        // Reset UI
        document.getElementById('calibration-actions').style.display = 'none';
        document.getElementById('set-angle-zero-btn').parentElement.style.display = 'flex';

        this.setControlsDisabled(false);
    }

    updateAxisLabelsFromType() {
        const xType = this.ui.xAxisType.value;
        const yType = this.ui.yAxisType.value;
        const zType = this.ui.zAxisType.value;

        const getLabel = (type) => {
            switch (type) {
                case 'angle': return 'Angle (deg)';
                case 'k': return 'k (1/Å)';
                case 'energy': return 'Energy (eV)';
                case 'hv': return 'Photon Energy (eV)';
                case 'scan': return 'Scan';
                default: return '';
            }
        };

        const labels = {
            x: getLabel(xType),
            y: getLabel(yType),
            z: getLabel(zType)
        };

        // Update Inputs
        document.getElementById('x-axis-label').value = labels.x;
        document.getElementById('y-axis-label').value = labels.y;
        document.getElementById('z-axis-label').value = labels.z;

        // Update Visualizer
        this.visualizer.setAxisLabels(labels);
    }

    displayMetadata(meta, filepath) {
        const container = document.getElementById('metadata-display');
        if (!container) return;

        // Create a clean object for display
        const displayObj = {
            "File Info": {
                "Path": filepath,
                "Shape": meta.data_info.shape,
                "Dimensions": meta.data_info.ndim,
                "Min/Max": [meta.data_info.min.toExponential(2), meta.data_info.max.toExponential(2)]
            },
            "File Attributes": meta.file_attributes || {},
            "IGOR Metadata": meta.igor_metadata || {},
            "Note": meta.note || ""
        };

        // Helper to format JSON nicely
        const jsonStr = JSON.stringify(displayObj, (key, value) => {
            // Filter out large arrays if any sneak in
            if (Array.isArray(value) && value.length > 10) {
                return `[Array(${value.length})]`;
            }
            return value;
        }, 2);

        container.textContent = jsonStr;
    }

    // ========== CROP MODE METHODS ==========

    startCrop() {
        if (!this.visualizer.data) {
            alert("Please load data first");
            return;
        }

        const success = this.visualizer.enterCropMode();
        if (!success) {
            alert("Failed to enter crop mode");
        }
    }

    confirmCrop() {
        const success = this.visualizer.applyCrop();
        if (success) {
            // Update App's source of truth with the cropped data
            this.originalData = this.visualizer.data;

            // Deep copy metadata
            const newMeta = JSON.parse(JSON.stringify(this.visualizer.metadata));

            // CRITICAL FIX for energy shift bug:
            // The problem: visualizer.metadata.axes have ALREADY been shifted by calibration offsets
            // (because recalculateData() subtracts the offsets from originalMeta before displaying).
            // When we crop, applyCrop2D slices these already-shifted axes.
            // If we save them as-is to originalMeta, and then recalculateData() subtracts offsets AGAIN,
            // we get double-shifting.
            //
            // Solution: Add the offsets BACK to the axes before saving to originalMeta.
            // This "un-shifts" them so they represent the original coordinate system.
            // Then recalculateData() will apply the offsets correctly.
            if (newMeta.axes.kx) {
                newMeta.axes.kx = newMeta.axes.kx.map(v => v + this.calibration.angleOffsetKx);
            }
            if (newMeta.axes.ky) {
                newMeta.axes.ky = newMeta.axes.ky.map(v => v + this.calibration.angleOffsetKy);
            }
            if (newMeta.axes.energy) {
                newMeta.axes.energy = newMeta.axes.energy.map(v => v + this.calibration.fermiOffset);
            }

            this.originalMeta = newMeta;

            // Update metadata display with new shape
            const filepath = this.selectedFilePath || "cropped data";
            this.displayMetadata(this.originalMeta, filepath);

            // Recalculate to ensure consistency
            this.recalculateData();

            // Auto-contrast for the new area
            this.visualizer.autoContrast();
            this.updateContrastUI();
        } else {
            alert("Failed to apply crop");
        }
    }

    cancelCrop() {
        this.visualizer.cancelCrop();
    }

    onCropModeEntered() {
        // Show crop action buttons
        if (this.ui.cropActions) {
            this.ui.cropActions.style.display = 'flex';
        }
        if (this.ui.cropBtn) {
            this.ui.cropBtn.style.display = 'none';
        }

        // Disable other controls
        this.setControlsDisabled(true);
    }

    onCropModeExited() {
        // Hide crop action buttons
        if (this.ui.cropActions) {
            this.ui.cropActions.style.display = 'none';
        }
        if (this.ui.cropBtn) {
            this.ui.cropBtn.style.display = 'block';
        }

        // Re-enable other controls
        this.setControlsDisabled(false);
    }

    setControlsDisabled(disabled) {
        const controls = [
            this.ui.colormapSelect,
            this.ui.cminSlider,
            this.ui.cmaxSlider,
            this.ui.autoContrastBtn,
            this.ui.resetContrastBtn,
            this.ui.xAxisType,
            this.ui.yAxisType,
            this.ui.zAxisType,
            this.ui.hvInput,
            this.ui.workFuncInput,
            this.ui.innerPotInput,
            this.ui.setAngleZeroBtn,
            this.ui.setFermiBtn,
            document.getElementById('angle-to-k-btn'),
            document.getElementById('reset-data-btn'),
            document.getElementById('export-btn'),
            document.getElementById('invert-colormap-check'),
            document.getElementById('mapping-mode-select'),
            document.getElementById('kz-mode-select'),
            // Enhancement controls
            this.ui.enhanceEnableCheck,
            this.ui.smoothSlider,
            this.ui.sharpenSlider,
            this.ui.bgSlider,
            this.ui.claheEnableCheck,
            this.ui.claheSlider
        ];

        controls.forEach(ctrl => {
            if (ctrl) {
                ctrl.disabled = disabled;
                if (disabled) {
                    ctrl.classList.add('crop-mode-disabled');
                } else {
                    ctrl.classList.remove('crop-mode-disabled');
                }
            }
        });
    }

    resetData() {
        if (!this.selectedFilePath && !this.selectedFileObj) {
            alert("No file loaded to reset");
            return;
        }

        // Reset calibration offsets
        this.calibration = { angleOffsetKx: 0, angleOffsetKy: 0, fermiOffset: 0 };

        // Reset axis type selectors to default
        this.ui.xAxisType.value = 'angle';
        this.ui.yAxisType.value = 'energy';
        if (this.ui.zAxisType) {
            this.ui.zAxisType.value = 'angle';
        }

        // Reset mapping UI
        const mappingSelect = document.getElementById('mapping-mode-select');
        const kzSelect = document.getElementById('kz-mode-select');
        const kzGroup = document.getElementById('kz-mode-group');
        if (mappingSelect) mappingSelect.value = 'kx-ky';
        if (kzSelect) kzSelect.value = 'convert';
        if (kzGroup) kzGroup.style.display = 'none';

        // Reload the file
        this.loadSelectedFile();
    }

    exportProfiles() {
        if (!this.originalData || !this.originalMeta) {
            alert("No data loaded");
            return;
        }

        const ndim = this.originalMeta.data_info.ndim;
        const shape = this.originalMeta.data_info.shape;
        const axes = this.visualizer.axes;
        const sliceIndices = this.visualizer.sliceIndices;

        // Get cursor positions
        const idxX = sliceIndices[1]; // Angle index
        const idxY = sliceIndices[0]; // Energy index
        const idxZ = sliceIndices[2]; // Scan index (for 3D)

        let content = "";

        if (ndim === 2) {
            const width = shape[1];  // Angle
            const height = shape[0]; // Energy

            // Extract EDC (vertical cut at cursor x position)
            const edc = [];
            for (let i = 0; i < height; i++) {
                edc.push(this.visualizer.data[i * width + idxX]);
            }

            // Extract MDC (horizontal cut at cursor y position)
            const mdc = [];
            const rowStart = idxY * width;
            for (let j = 0; j < width; j++) {
                mdc.push(this.visualizer.data[rowStart + j]);
            }

            // Build TSV content
            content = "# EDC and MDC Profiles\n";
            content += `# Cursor position: Angle=${axes.kx[idxX].toFixed(4)}, Energy=${axes.energy[idxY].toFixed(4)}\n`;
            content += "#\n";
            content += "# EDC (Energy vs Intensity at fixed Angle)\n";
            content += "Energy(eV)\tIntensity\n";
            for (let i = 0; i < height; i++) {
                content += `${axes.energy[i].toFixed(6)}\t${edc[i].toExponential(6)}\n`;
            }
            content += "\n# MDC (Angle vs Intensity at fixed Energy)\n";
            content += "Angle(deg)\tIntensity\n";
            for (let j = 0; j < width; j++) {
                content += `${axes.kx[j].toFixed(6)}\t${mdc[j].toExponential(6)}\n`;
            }

        } else if (ndim === 3) {
            const width = shape[1];  // Angle
            const height = shape[0]; // Energy
            const depth = shape[2];  // Scan

            // Extract XY slice for EDC and MDC
            const sliceXY = this.visualizer.extractSlice(2, idxZ);

            // Extract EDC from XY slice
            const edc = [];
            for (let i = 0; i < height; i++) {
                edc.push(sliceXY[i * width + idxX]);
            }

            // Extract MDC from XY slice
            const mdc = [];
            const rowStart = idxY * width;
            for (let j = 0; j < width; j++) {
                mdc.push(sliceXY[rowStart + j]);
            }

            // Extract Scan profile
            const scanProfile = [];
            for (let k = 0; k < depth; k++) {
                const idx = idxY * (width * depth) + idxX * depth + k;
                scanProfile.push(this.visualizer.data[idx]);
            }

            // Build TSV content
            content = "# EDC, MDC, and Scan Profiles\n";
            content += `# Cursor position: Angle=${axes.kx[idxX].toFixed(4)}, Energy=${axes.energy[idxY].toFixed(4)}, Scan=${axes.ky[idxZ].toFixed(4)}\n`;
            content += "#\n";
            content += "# EDC (Energy vs Intensity at fixed Angle and Scan)\n";
            content += "Energy(eV)\tIntensity\n";
            for (let i = 0; i < height; i++) {
                content += `${axes.energy[i].toFixed(6)}\t${edc[i].toExponential(6)}\n`;
            }
            content += "\n# MDC (Angle vs Intensity at fixed Energy and Scan)\n";
            content += "Angle(deg)\tIntensity\n";
            for (let j = 0; j < width; j++) {
                content += `${axes.kx[j].toFixed(6)}\t${mdc[j].toExponential(6)}\n`;
            }
            content += "\n# Scan Profile (Scan vs Intensity at fixed Angle and Energy)\n";
            content += "Scan\tIntensity\n";
            for (let k = 0; k < depth; k++) {
                content += `${axes.ky[k].toFixed(6)}\t${scanProfile[k].toExponential(6)}\n`;
            }
        }

        // Download file
        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'profiles.tsv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
