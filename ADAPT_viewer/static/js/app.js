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
            alignThetaBtn: document.getElementById('align-theta-btn'),

            // hv Mapping controls
            hvMappingToggleHeader: document.getElementById('hv-mapping-toggle-header'),
            hvMappingControls: document.getElementById('hv-mapping-mode'),
            hvMappingHeaderIcon: document.getElementById('hv-mapping-header-icon'),
            hvMappingEnableCheck: document.getElementById('hv-mapping-enable-check'),
            efFitRangeInput: document.getElementById('ef-fit-range-input'),
            normalizationBtn: document.getElementById('normalization-btn'),
            convertKxKzBtn: document.getElementById('convert-kxkz-btn'),
            convertKxHvBtn: document.getElementById('convert-kxhv-btn'),
            hvMappingConditionalContent: document.getElementById('hv-mapping-conditional-content'),

            // New Toggles
            colormapToggleHeader: document.getElementById('colormap-toggle-header'),
            colormapControls: document.getElementById('colormap-controls'),
            colormapHeaderIcon: document.getElementById('colormap-header-icon'),
            parametersToggleHeader: document.getElementById('parameters-toggle-header'),
            parametersControls: document.getElementById('parameters-controls'),
            parametersHeaderIcon: document.getElementById('parameters-header-icon'),

            // Page Switcher UI
            pageImageBtn: document.getElementById('page-image-btn'),
            pageBZBtn: document.getElementById('page-bz-btn'),
            pageImageControls: document.getElementById('page-image-controls'),
            pageBZControls: document.getElementById('page-bz-controls'),
            plotContainer: document.getElementById('plot-container'),
            bzContainer: document.getElementById('bz-container'),

            // BZ UI
            bzInputManualBtn: document.getElementById('bz-input-manual-btn'),
            bzInputMPBtn: document.getElementById('bz-input-mp-btn'),
            bzManualInputs: document.getElementById('bz-manual-inputs'),
            bzMPInputs: document.getElementById('bz-mp-inputs'),
            bzLatticeParams: document.getElementById('bz-lattice-params'),
            bzLatticeAngles: document.getElementById('bz-lattice-angles'),
            bzCrystalName: document.getElementById('bz-crystal-name'),
            bzCrystalName: document.getElementById('bz-crystal-name'),
            bzMPId: document.getElementById('bz-mp-id'),
            bzCalculateBtn: document.getElementById('bz-calculate-btn'),

            // New Group Headers (Reorganization)
            visualizationGroupHeader: document.getElementById('visualization-group-header'),
            visualizationGroupIcon: document.getElementById('visualization-group-icon'),
            visualizationGroupContent: document.getElementById('visualization-group-content'),

            processingGroupHeader: document.getElementById('processing-group-header'),
            processingGroupIcon: document.getElementById('processing-group-icon'),
            processingGroupContent: document.getElementById('processing-group-content'),

            exportGroupHeader: document.getElementById('export-group-header'),
            exportGroupIcon: document.getElementById('export-group-icon'),
            exportGroupContent: document.getElementById('export-group-content'),

            themeToggle: document.getElementById('theme-toggle')
        };

        this.init();
        this.initTheme();
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

        // Sync initial enhancement state
        setTimeout(() => {
            this.updateEnhancement();
        }, 100);

        // Sync HV Mapping state
        setTimeout(() => {
            if (this.ui.hvMappingEnableCheck && this.ui.hvMappingEnableCheck.checked) {
                this.ui.hvMappingEnableCheck.dispatchEvent(new Event('change'));
            }
        }, 100);


        // Check for session parameter from ADAPT Browser integration
        await this.checkForSession();
    }


    switchPage(page) {
        if (page === 'image') {
            this.ui.pageImageBtn.classList.replace('secondary-btn', 'primary-btn');
            this.ui.pageBZBtn.classList.replace('primary-btn', 'secondary-btn');

            this.ui.pageImageControls.style.display = 'block';
            this.ui.pageBZControls.style.display = 'none';

            this.ui.plotContainer.style.display = 'flex';
            this.ui.bzContainer.style.display = 'none';

            // Trigger resize to fix canvas size if it was hidden
            window.dispatchEvent(new Event('resize'));
        } else {
            this.ui.pageBZBtn.classList.replace('secondary-btn', 'primary-btn');
            this.ui.pageImageBtn.classList.replace('primary-btn', 'secondary-btn');

            this.ui.pageBzControls ? this.ui.pageBzControls.style.display = 'block' : this.ui.pageBZControls.style.display = 'block'; // Case safety
            this.ui.pageImageControls.style.display = 'none';

            this.ui.bzContainer.style.display = 'flex';
            this.ui.plotContainer.style.display = 'none';
        }
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
            return;
        }

        try {
            // Show loading state
            if (this.ui.dataInfo) {
                this.ui.dataInfo.textContent = "Loading from ADAPT Browser...";
            }

            // Fetch session data from backend
            const sessionData = await this.dataLoader.loadSession();

            if (!sessionData || !sessionData.file_path) {
                if (this.ui.dataInfo) {
                    this.ui.dataInfo.textContent = "No session file found. Please select a file.";
                }
                return;
            }

            // Set the file path and trigger load
            const filePath = sessionData.file_path;

            // Store the path and load the file
            this.selectedFilePath = filePath;

            // Update UI with file info
            const basename = filePath.split(/[/\\]/).pop();
            document.getElementById('selected-file-label').textContent = basename;

            // Load the file
            await this.loadSelectedFile();

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

    initTheme() {
        // Check localStorage or system preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            this.setTheme(savedTheme);
        } else {
            // Default to dark, but check system preference optionally
            // const prefersLight = window.matchMedia('(prefers-color-scheme: light)').matches;
            // this.setTheme(prefersLight ? 'light' : 'dark');
            this.setTheme('dark'); // Default for now
        }
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
    }

    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);

        // Update Toggle Icon
        if (this.ui.themeToggle) {
            if (theme === 'light') {
                // Sun icon
                this.ui.themeToggle.innerHTML = `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="5"></circle>
                        <line x1="12" y1="1" x2="12" y2="3"></line>
                        <line x1="12" y1="21" x2="12" y2="23"></line>
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                        <line x1="1" y1="12" x2="3" y2="12"></line>
                        <line x1="21" y1="12" x2="23" y2="12"></line>
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                    </svg>
                `;
            } else {
                // Moon icon
                this.ui.themeToggle.innerHTML = `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                    </svg>
                `;
            }
        }

        // Trigger redraw if data loaded, as some canvas elements use theme colors
        if (this.visualizer && this.visualizer.data) {
            this.visualizer.draw();
        }
    }

    setupEventListeners() {
        // Theme Toggle
        if (this.ui.themeToggle) {
            this.ui.themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        // Toggle Logic for Toolbar Groups
        const setupToggle = (header, content, icon) => {
            if (!header || !content) return;
            // Set initial state based on current display style or default
            // But simply adding listener is enough for interaction.

            header.addEventListener('click', () => {
                // Check computed style if inline style is not set
                const isVisible = content.style.display !== 'none';

                content.style.display = isVisible ? 'none' : 'block';
                if (icon) {
                    // Open -> Close (0deg), Closed -> Open (180deg)
                    icon.style.transform = isVisible ? 'rotate(0deg)' : 'rotate(180deg)';
                }
            });
        };

        // New Groups
        setupToggle(this.ui.visualizationGroupHeader, this.ui.visualizationGroupContent, this.ui.visualizationGroupIcon);
        setupToggle(this.ui.processingGroupHeader, this.ui.processingGroupContent, this.ui.processingGroupIcon);
        setupToggle(this.ui.exportGroupHeader, this.ui.exportGroupContent, this.ui.exportGroupIcon);

        // Existing Sections (re-using new logic to be safe, or just adding on top?)
        // The existing sections might not have listeners if I removed them? 
        // I haven't removed any JS code yet, only added. 
        // So if there was existing logic elsewhere, it might duplicate.
        // But I didn't see explicit listeners for colormapToggleHeader in the file earlier.
        // I will add them here to be sure.
        // setupToggle(this.ui.colormapToggleHeader, this.ui.colormapControls, this.ui.colormapHeaderIcon);
        // setupToggle(this.ui.enhanceToggleHeader, this.ui.enhanceControls, this.ui.enhanceHeaderIcon);
        // setupToggle(this.ui.parametersToggleHeader, this.ui.parametersControls, this.ui.parametersHeaderIcon);
        // Alignment doesn't have a header in old code, but now it's inside Processing.
        // HV Mapping has a header.
        setupToggle(this.ui.hvMappingToggleHeader, this.ui.hvMappingControls, this.ui.hvMappingHeaderIcon);

        // Page Switching
        if (this.ui.pageImageBtn) {
            this.ui.pageImageBtn.addEventListener('click', () => this.switchPage('image'));
            this.ui.pageBZBtn.addEventListener('click', () => this.switchPage('bz'));
        }

        // BZ Input Method Toggle
        if (this.ui.bzInputManualBtn) {
            this.ui.bzInputManualBtn.addEventListener('click', () => {
                this.ui.bzInputManualBtn.classList.replace('secondary-btn', 'primary-btn');
                this.ui.bzInputManualBtn.style.color = 'white';
                this.ui.bzInputMPBtn.classList.replace('primary-btn', 'secondary-btn');
                this.ui.bzInputMPBtn.style.color = 'var(--text-secondary)';

                this.ui.bzManualInputs.style.display = 'block';
                this.ui.bzMPInputs.style.display = 'none';
            });
            this.ui.bzInputMPBtn.addEventListener('click', () => {
                this.ui.bzInputMPBtn.classList.replace('secondary-btn', 'primary-btn');
                this.ui.bzInputMPBtn.style.color = 'white';
                this.ui.bzInputManualBtn.classList.replace('primary-btn', 'secondary-btn');
                this.ui.bzInputManualBtn.style.color = 'var(--text-secondary)';

                this.ui.bzManualInputs.style.display = 'none';
                this.ui.bzMPInputs.style.display = 'block';
            });
        }


        // Calculate BZ
        if (this.ui.bzCalculateBtn) {
            this.ui.bzCalculateBtn.addEventListener('click', async () => {
                const isManual = this.ui.bzManualInputs.style.display !== 'none';
                let payload = { method: isManual ? 'manual' : 'mp' };

                if (isManual) {
                    const paramsStr = this.ui.bzLatticeParams.value || "3.5, 3.5, 3.5";
                    const anglesStr = this.ui.bzLatticeAngles.value || "90, 90, 90";
                    const params = paramsStr.split(',').map(s => parseFloat(s.trim()));
                    const angles = anglesStr.split(',').map(s => parseFloat(s.trim()));

                    if (params.length !== 3 || params.some(isNaN)) {
                        alert("Invalid lattice parameters. Format: a, b, c"); return;
                    }
                    if (angles.length !== 3 || angles.some(isNaN)) {
                        alert("Invalid angles. Format: alpha, beta, gamma"); return;
                    }
                    payload.a = params[0]; payload.b = params[1]; payload.c = params[2];
                    payload.alpha = angles[0]; payload.beta = angles[1]; payload.gamma = angles[2];
                    payload.crystal_name = this.ui.bzCrystalName.value;
                } else {
                    const mpId = this.ui.bzMPId.value;
                    if (!mpId) { alert("Please enter MP ID"); return; }
                    payload.mp_id = mpId;
                }

                // Detect Theme
                const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
                payload.theme = currentTheme;

                // Read intersection plane parameters (optional)
                const millerInput = document.getElementById('bz-plane-miller');
                const distanceInput = document.getElementById('bz-plane-distance');
                const colorInput = document.getElementById('bz-plane-color');

                if (millerInput && millerInput.value.trim()) {
                    const millerStr = millerInput.value.trim();
                    const millerParts = millerStr.split(',').map(s => parseFloat(s.trim()));
                    if (millerParts.length === 3 && !millerParts.some(isNaN)) {
                        payload.plane_miller = millerParts;
                        payload.plane_distance = distanceInput ? parseFloat(distanceInput.value) || 0.0 : 0.0;
                        payload.plane_color = colorInput ? colorInput.value : '#00ff00';
                    }
                }

                try {
                    this.ui.bzCalculateBtn.disabled = true;
                    this.ui.bzCalculateBtn.textContent = "Calculating...";


                    const res = await fetch('/api/process/brillouin_zone', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || "Failed");
                    }

                    const contentType = res.headers.get('content-type');

                    if (contentType && contentType.includes('application/json')) {
                        // Plotly JSON response
                        const plotlyData = await res.json();

                        // Clear container
                        this.ui.bzContainer.innerHTML = '';
                        // Set background based on theme strictly to correct previous issues
                        // Dark: #121212, Light: #ffffff
                        this.ui.bzContainer.style.background = currentTheme === 'dark' ? '#121212' : '#ffffff';
                        this.ui.bzContainer.style.display = 'block'; // Block for Plotly div

                        // Ensure unique ID for Plotly
                        const plotlyDivId = 'bz-plotly-div';
                        let plotlyDiv = document.getElementById(plotlyDivId);
                        if (!plotlyDiv) {
                            plotlyDiv = document.createElement('div');
                            plotlyDiv.id = plotlyDivId;
                            plotlyDiv.style.width = '100%';
                            plotlyDiv.style.height = '100%';
                            this.ui.bzContainer.appendChild(plotlyDiv);
                        }

                        // Render Plotly
                        if (window.Plotly) {
                            Plotly.newPlot(plotlyDivId, plotlyData.data, plotlyData.layout, { responsive: true });
                        } else {
                            throw new Error("Plotly library not loaded");
                        }

                    } else {
                        // Fallback to Image (Blob) response
                        const blob = await res.blob();
                        const url = URL.createObjectURL(blob);

                        // Render in bzContainer
                        this.ui.bzContainer.innerHTML = '';
                        const img = document.createElement('img');
                        img.src = url;
                        img.style.maxWidth = '100%';
                        img.style.maxHeight = '100%';
                        img.style.objectFit = 'contain';
                        this.ui.bzContainer.appendChild(img);
                        this.ui.bzContainer.style.background = currentTheme === 'dark' ? '#121212' : '#ffffff';; // Ensure background matches theme
                        this.ui.bzContainer.style.display = 'flex';
                        this.ui.bzContainer.style.justifyContent = 'center';
                        this.ui.bzContainer.style.alignItems = 'center';
                    }

                } catch (e) {
                    console.error(e);
                    alert("Error calculation BZ: " + e.message);
                } finally {
                    this.ui.bzCalculateBtn.disabled = false;
                    this.ui.bzCalculateBtn.textContent = "Calculate BZ";
                }
            });
        }

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

        document.getElementById('angle-to-k-btn').addEventListener('click', async () => {
            // Check if hv is provided
            const hvInput = document.getElementById('hv-input');
            const hv = parseFloat(hvInput.value);
            if (isNaN(hv) || hv <= 0) {
                alert("Please provide a valid Photon Energy (hv) for conversion.");
                return;
            }

            const btn = document.getElementById('angle-to-k-btn');
            btn.disabled = true;
            btn.textContent = "Converting...";
            const restoreBtn = () => {
                btn.disabled = false;
                btn.textContent = "Convert to k";
            };

            try {
                const workFunc = parseFloat(this.ui.workFuncInput.value) || 4.5;
                const innerPot = parseFloat(this.ui.innerPotInput.value) || 12.57;

                // Check simple HV mapping check
                // For proper logic, we might need to check if it's 3D and mapping is enabled
                const hvMappingCheck = document.getElementById('hv-mapping-enable-check');
                const isHvMappingEnabled = hvMappingCheck ? hvMappingCheck.checked : false;

                const res = await fetch('/api/process/convert_k', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        path: this.selectedFilePath,
                        hv: hv,
                        work_function: workFunc,
                        inner_potential: innerPot,
                        hv_mapping_enabled: isHvMappingEnabled
                    })
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Conversion failed");
                }

                const data = await res.json();

                // Update file path and load
                this.selectedFilePath = data.filename;
                await this.loadSelectedFile();

                // Set relevant UI state
                this.ui.xAxisType.value = 'k';
                if (this.ui.dataInfo) {
                    this.ui.dataInfo.textContent += " (Converted to k-space)";
                }

            } catch (e) {
                console.error("Conversion Error:", e);
                alert("Conversion Error: " + e.message);
            } finally {
                restoreBtn();
            }
        });

        // Convert Angle to KX-HV (HV Scan) button
        const convertKxHvBtn = document.getElementById('convert-kxhv-btn');
        if (convertKxHvBtn) {
            convertKxHvBtn.addEventListener('click', async () => {
                const btn = convertKxHvBtn;
                btn.disabled = true;
                btn.textContent = "Converting...";
                const restoreBtn = () => {
                    btn.disabled = false;
                    btn.textContent = "Convert to kx-hv";
                };

                try {
                    const workFunc = parseFloat(this.ui.workFuncInput.value) || 4.5;
                    const innerPot = parseFloat(this.ui.innerPotInput.value) || 12.57;

                    const res = await fetch('/api/process/convert_k', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            path: this.selectedFilePath,
                            hv: null, // Not needed for hv scan
                            work_function: workFunc,
                            inner_potential: innerPot,
                            hv_mapping_enabled: true,
                            is_hv_scan: true,
                            hv_dim: 'scan',
                            convert_hv_to_kz: false
                        })
                    });

                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || "Conversion failed");
                    }

                    const data = await res.json();

                    // Update file path and load
                    this.selectedFilePath = data.filename;
                    await this.loadSelectedFile();

                    // Set relevant UI state
                    this.ui.xAxisType.value = 'k'; // or kx
                    if (this.ui.dataInfo) {
                        this.ui.dataInfo.textContent += " (Converted to kx-hv)";
                    }

                } catch (e) {
                    console.error("Conversion Error:", e);
                    alert("Conversion Error: " + e.message);
                } finally {
                    restoreBtn();
                }
            });
        }

        // Convert Angle to KX-KZ (HV Scan -> Kz) button
        const convertKxKzBtn = document.getElementById('convert-kxkz-btn');
        if (convertKxKzBtn) {
            convertKxKzBtn.addEventListener('click', async () => {
                const btn = convertKxKzBtn;
                btn.disabled = true;
                btn.textContent = "Converting...";
                const restoreBtn = () => {
                    btn.disabled = false;
                    btn.textContent = "Convert to kx-kz";
                };

                try {
                    const workFunc = parseFloat(this.ui.workFuncInput.value) || 4.5;
                    const innerPot = parseFloat(this.ui.innerPotInput.value) || 12.57;

                    const res = await fetch('/api/process/convert_k', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            path: this.selectedFilePath,
                            hv: null,
                            work_function: workFunc,
                            inner_potential: innerPot,
                            hv_mapping_enabled: true,
                            is_hv_scan: true,
                            hv_dim: 'scan',
                            convert_hv_to_kz: true
                        })
                    });

                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || "Conversion failed");
                    }

                    const data = await res.json();

                    this.selectedFilePath = data.filename;
                    await this.loadSelectedFile();

                    this.ui.xAxisType.value = 'k';
                    if (this.ui.zAxisType) this.ui.zAxisType.value = 'k'; // kz usually maps to 'scan' dim, but we might want to call it k?
                    // Actually, convert_angle_to_k for kz usually names the axis 'kz'.
                    // The loader maps 'kz' to 'ky' in normalized axes if 'ky' is present? No, let's check loader.

                    if (this.ui.dataInfo) {
                        this.ui.dataInfo.textContent += " (Converted to kx-kz)";
                    }

                } catch (e) {
                    console.error("Conversion Error:", e);
                    alert("Conversion Error: " + e.message);
                } finally {
                    restoreBtn();
                }
            });
        }
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

        // Colormap Controls Toggle
        if (this.ui.colormapToggleHeader) {
            // Check initial state
            if (this.ui.colormapControls && this.ui.colormapControls.style.display === 'none') {
                this.ui.colormapHeaderIcon.classList.remove('rotate-icon');
            }

            this.ui.colormapToggleHeader.addEventListener('click', () => {
                const isHidden = this.ui.colormapControls.style.display === 'none';
                this.ui.colormapControls.style.display = isHidden ? 'block' : 'none';
                this.ui.colormapHeaderIcon.classList.toggle('rotate-icon', isHidden);
            });
        }

        // Parameters Controls Toggle
        if (this.ui.parametersToggleHeader) {
            // Check initial state
            if (this.ui.parametersControls && this.ui.parametersControls.style.display === 'none') {
                this.ui.parametersHeaderIcon.classList.remove('rotate-icon');
            }

            this.ui.parametersToggleHeader.addEventListener('click', () => {
                const isHidden = this.ui.parametersControls.style.display === 'none';
                this.ui.parametersControls.style.display = isHidden ? 'block' : 'none';
                this.ui.parametersHeaderIcon.classList.toggle('rotate-icon', isHidden);
            });
        }

        this.updateEnhancement = () => {
            const claheEnabled = this.ui.claheEnableCheck ? this.ui.claheEnableCheck.checked : false;
            const curvatureEnabled = this.ui.curvatureEnableCheck ? this.ui.curvatureEnableCheck.checked : false;

            // Sub-features imply global enabled
            const isEnabled = claheEnabled || curvatureEnabled ||
                (parseFloat(this.ui.smoothSlider.value) > 0) ||
                (parseFloat(this.ui.sharpenSlider.value) > 0) ||
                (parseInt(this.ui.bgSlider.value) > 0);

            const opts = {
                enabled: isEnabled,
                smoothing: parseFloat(this.ui.smoothSlider.value),
                sharpen: parseFloat(this.ui.sharpenSlider.value),
                background: parseInt(this.ui.bgSlider.value),
                clahe: claheEnabled,
                claheClip: parseFloat(this.ui.claheSlider.value),
                curvature: curvatureEnabled,
                curvatureStrength: this.ui.curvatureSlider ? parseFloat(this.ui.curvatureSlider.value) : 1.0
            };
            this.visualizer.setEnhancement(opts);
        };

        // this.ui.enhanceEnableCheck removed.


        // hv Mapping Controls


        if (this.ui.hvMappingEnableCheck) {
            this.ui.hvMappingEnableCheck.addEventListener('change', (e) => {
                // Toggle specific buttons visibility
                const normBtn = document.getElementById('normalization-btn');
                const kxkzBtn = document.getElementById('convert-kxkz-btn');
                const kxhvBtn = document.getElementById('convert-kxhv-btn');
                const angleToKBtn = document.getElementById('angle-to-k-btn');

                if (normBtn) normBtn.style.display = e.target.checked ? 'block' : 'none';
                // if (kxkzBtn) kxkzBtn.style.display = e.target.checked ? 'block' : 'none'; // Temporarily hidden as functionality is incomplete
                if (kxhvBtn) kxhvBtn.style.display = e.target.checked ? 'block' : 'none';

                // Toggle standard convert button inversely? 
                // The user request didn't explicitly ask to hide angle-to-k-btn, but typically mode switches imply substitution.
                // Previously the entire convert section was hidden. Now we keep it for new buttons.
                // Let's hide the standard button to be clean.
                if (angleToKBtn) angleToKBtn.style.display = e.target.checked ? 'none' : 'block';

                if (this.ui.hvMappingConditionalContent) {
                    this.ui.hvMappingConditionalContent.style.display = e.target.checked ? 'block' : 'none';
                }

                // Toggle EF Fit Range
                const efFitRangeRow = document.getElementById('ef-fit-range-row');
                if (efFitRangeRow) {
                    efFitRangeRow.style.display = e.target.checked ? 'flex' : 'none';
                }
            });
        }

        if (this.ui.normalizationBtn) {
            this.ui.normalizationBtn.addEventListener('click', () => {
                this.normalize3DScans();
            });
        }

        // Note: Auto-align checkbox and re-detect button removed; use "Auto Set EF" in Convert controls.

        this.ui.smoothSlider.addEventListener('input', (e) => {
            if (this.ui.smoothVal) this.ui.smoothVal.textContent = e.target.value;
            this.updateEnhancement();
        });

        this.ui.sharpenSlider.addEventListener('input', (e) => {
            if (this.ui.sharpenVal) this.ui.sharpenVal.textContent = e.target.value;
            this.updateEnhancement();
        });

        this.ui.bgSlider.addEventListener('input', (e) => {
            if (this.ui.bgVal) this.ui.bgVal.textContent = e.target.value;
            this.updateEnhancement();
        });

        this.ui.claheEnableCheck.addEventListener('change', (e) => {
            this.ui.claheControls.style.display = e.target.checked ? 'block' : 'none';
            this.updateEnhancement();
        });

        this.ui.claheSlider.addEventListener('input', (e) => {
            this.ui.claheVal.textContent = e.target.value;
            this.updateEnhancement();
        });

        // Curvature controls
        if (this.ui.curvatureEnableCheck) {
            this.ui.curvatureEnableCheck.addEventListener('change', (e) => {
                if (this.ui.curvatureControls) {
                    this.ui.curvatureControls.style.display = e.target.checked ? 'block' : 'none';
                }
                if (this.ui.curvatureControls) {
                    this.ui.curvatureControls.style.display = e.target.checked ? 'block' : 'none';
                }
                this.updateEnhancement();
            });
        }

        if (this.ui.curvatureSlider) {
            this.ui.curvatureSlider.addEventListener('input', (e) => {
                if (this.ui.curvatureVal) this.ui.curvatureVal.textContent = e.target.value;
                this.updateEnhancement();
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

        // Listen for changes in inputs to update cursor
        [this.ui.efInput, this.ui.thetaInput, this.ui.scanInput].forEach(input => {
            if (input) {
                input.addEventListener('input', () => {
                    const ef = parseFloat(this.ui.efInput.value);
                    const theta = parseFloat(this.ui.thetaInput.value);
                    const scan = this.ui.scanInput ? parseFloat(this.ui.scanInput.value) : undefined;

                    // Construct update object only with valid numbers
                    const update = {};
                    if (!isNaN(ef)) update.y = ef;
                    if (!isNaN(theta)) update.x = theta;
                    if (!isNaN(scan)) update.z = scan;

                    this.visualizer.setCursor(update);
                });
            }
        });

        // Align Energy button - align energy axis to EF value
        if (this.ui.alignEnergyBtn) {
            this.ui.alignEnergyBtn.addEventListener('click', async () => {
                let ef = parseFloat(this.ui.efInput.value) || 0;

                // Show loading state
                this.ui.alignEnergyBtn.disabled = true;
                this.ui.alignEnergyBtn.textContent = "Aligning...";
                const restoreBtn = () => {
                    this.ui.alignEnergyBtn.disabled = false;
                    this.ui.alignEnergyBtn.textContent = "Align Energy";
                };

                try {
                    // Prepare parameters
                    const hvMappingEnabled = this.ui.hvMappingEnableCheck && this.ui.hvMappingEnableCheck.checked;
                    let fitRange = 0.5;

                    if (hvMappingEnabled) {
                        fitRange = parseFloat(this.ui.efFitRangeInput.value) || 0.5;
                    }

                    // Call Align API
                    // For 3D data with hv mapping, we send hv_mapping_enabled=true and fit_range
                    // The backend handles the slice-by-fit logic.
                    // For 2D data or 3D without mapping, it behaves as a simple shift.

                    const alignRes = await fetch('/api/process/align', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            path: this.selectedFilePath,
                            axis: 'energy',
                            offset: ef,
                            method: `fermi_alignment (E_F=${ef.toFixed(4)})`,
                            hv_mapping_enabled: hvMappingEnabled,
                            fit_range: fitRange
                        })
                    }).then(r => r.json());

                    if (alignRes.filename) {
                        // Reload with new file
                        this.selectedFilePath = alignRes.filename;
                        await this.loadSelectedFile({ keepSettings: true });

                        // Reset EF input to 0 since we act as if it is now aligned
                        this.ui.efInput.value = "0.0000";
                        // Reset visual calibration offsets since data is now shifted
                        this.calibration.fermiOffset = 0;
                        this.recalculateData();

                        if (this.ui.dataInfo) this.ui.dataInfo.textContent = hvMappingEnabled ?
                            "Energy Aligned (3D Fit & Resample)" : "Energy Aligned (Constant Shift)";
                    } else {
                        throw new Error(alignRes.detail || "Alignment API returned no filename");
                    }

                } catch (e) {
                    console.error("Align Error:", e);
                    alert("Align Error: " + e.message);
                } finally {
                    restoreBtn();
                }
            });
        }

        // Align Theta button
        if (this.ui.alignThetaBtn) {
            this.ui.alignThetaBtn.addEventListener('click', async () => {
                const theta = parseFloat(this.ui.thetaInput.value) || 0;
                let scan = 0;
                if (this.ui.scanInput) {
                    scan = parseFloat(this.ui.scanInput.value) || 0;
                }

                // Check HV Mapping status
                const hvMappingCheck = document.getElementById('hv-mapping-enable-check');
                const isHvMappingEnabled = hvMappingCheck ? hvMappingCheck.checked : false;

                this.ui.alignThetaBtn.disabled = true;
                this.ui.alignThetaBtn.textContent = "Aligning...";
                const restoreBtn = () => {
                    this.ui.alignThetaBtn.disabled = false;
                    this.ui.alignThetaBtn.textContent = "Align Theta";
                };

                try {
                    let currentPath = this.selectedFilePath;

                    // Unified Alignment Request
                    const res = await fetch('/api/process/align', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            path: currentPath,
                            axis: 'kx', // Primary axis (usually mapped to angle/theta/kx)
                            offset: theta,
                            scan_axis: 'scan', // Secondary axis
                            scan_offset: scan,
                            hv_mapping_enabled: isHvMappingEnabled,
                            method: 'composite_align'
                        })
                    });

                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || "Align failed");
                    }

                    const data = await res.json();
                    currentPath = data.filename;

                    // Reload
                    this.selectedFilePath = currentPath;
                    await this.loadSelectedFile({ keepSettings: true });

                    // Reset inputs
                    this.ui.thetaInput.value = "0.00";
                    if (this.ui.scanInput) this.ui.scanInput.value = "0.00";

                    // Reset visual offsets
                    this.calibration.angleOffsetKx = 0;
                    this.calibration.angleOffsetKy = 0;
                    this.recalculateData();

                } catch (e) {
                    console.error("Align Error:", e);
                    alert("Align Error: " + e.message);
                } finally {
                    restoreBtn();
                }
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

    async loadSelectedFile(options = {}) {
        let filename = this.selectedFilePath;
        if (!filename && !this.selectedFileObj) return;

        const keepSettings = options.keepSettings || false;

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

            // Reset calibration unless keeping settings
            if (!keepSettings) {
                this.calibration = { angleOffsetKx: 0, angleOffsetKy: 0, fermiOffset: 0 };
            }

            // 1. Load Metadata
            const meta = await this.dataLoader.loadMetadata(filename);
            this.originalMeta = meta;

            // 2. Load Binary Data
            const data = await this.dataLoader.loadData(filename);
            this.originalData = data;

            // 3. Initial Recalculate (handles display)


            // Settings regarding HV Mapping, CLAHE, Curvature are now preserved throughout file loads.


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

            // Handle HV Mapping Checkbox for 2D/3D
            // Handle HV Mapping Checkbox for 2D/3D
            const hvCheck = document.getElementById('hv-mapping-enable-check');
            if (hvCheck) {
                // Find proper container: ideally #hv-mapping-mode, else checkbox-row
                const container = document.getElementById('hv-mapping-mode') || hvCheck.closest('.checkbox-row');

                if (!is3D) {
                    // 2D Data: Hide completely
                    hvCheck.checked = false;
                    if (container) container.style.display = 'none';

                    // Also hide conditional content
                    const hvContent = document.getElementById('hv-mapping-conditional-content');
                    if (hvContent) hvContent.style.display = 'none';
                } else {
                    // 3D Data: Show
                    if (container) container.style.display = 'block'; // or flex/initial
                }
            }

            // Show/hide scan input row for 3D data
            if (this.ui.scanInputRow) {
                this.ui.scanInputRow.style.display = is3D ? 'flex' : 'none';
            }

            // Default axis assignment: for 3D data, set X to Angle, Y to Energy, Z to Scan.
            // User can change these via the Axis Management panel; reflect defaults in the UI.
            // Default axis assignment based on metadata
            let defaultX = 'angle';
            let defaultY = 'energy';
            let defaultZ = 'scan';

            // Check if data is converted to k-space
            const conversion = (meta.metadata && meta.metadata.conversion) || '';
            const isKSpace = conversion.includes('to_k') || conversion.includes('kx_ky') || (meta.axes && meta.axes.k);

            if (isKSpace) {
                defaultX = 'k';
            }

            // Apply defaults
            if (this.ui.xAxisType) this.ui.xAxisType.value = defaultX;
            if (this.ui.yAxisType) this.ui.yAxisType.value = defaultY;

            if (is3D) {
                if (this.ui.zAxisType) this.ui.zAxisType.value = defaultZ;
            }
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

    async confirmCrop() {
        if (!this.visualizer.cropBox) return;

        const indices = this.visualizer.getCropIndices();
        if (!indices) {
            alert("Could not determine crop indices");
            return;
        }

        const { x0, x1, y0, y1 } = indices;
        const ranges = {};

        // Map visualizer view coordinates to data dimensions (x=Angle, y=Energy, z=Scan)
        if (this.originalMeta.data_info.ndim === 2) {
            // 2D: X(dim1)=Angle, Y(dim0)=Energy
            ranges.x = [x0, x1];
            ranges.y = [y0, y1];
        } else {
            // 3D mapping based on active view
            const plane = this.visualizer.activeCropView;
            if (plane === 'xy') {
                // X=Angle(dim1), Y=Energy(dim0)
                ranges.x = [x0, x1];
                ranges.y = [y0, y1];
            } else if (plane === 'xz') {
                // X=Scan(dim2), Y=Energy(dim0)
                ranges.z = [x0, x1];
                ranges.y = [y0, y1];
            } else if (plane === 'yz') {
                // X=Scan(dim2), Y=Angle(dim1)
                ranges.z = [x0, x1];
                ranges.x = [y0, y1];
            } else {
                alert("Unknown crop view");
                return;
            }
        }

        // UI State
        if (this.ui.confirmCropBtn) {
            this.ui.confirmCropBtn.disabled = true;
            this.ui.confirmCropBtn.textContent = "Cropping...";
        }

        try {
            const res = await fetch('/api/process/crop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    path: this.selectedFilePath,
                    ranges: ranges
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Crop failed");
            }

            const data = await res.json();

            // Exit local crop mode UI
            this.visualizer.exitCropMode();

            // Update file path and reload
            this.selectedFilePath = data.filename;

            // Reload with keepSettings=true to preserve viewer settings (colormap etc)
            // But NOT calibration if the axes have changed? 
            // Actually, keepSettings=true preserves calibration offsets.
            // Since the new file will have the same PHYSICAL values (just fewer of them),
            // and calibration offsets are physical shifts, they should remain valid.
            // E.g. Fermi is still at -0.5eV if we didn't crop it out.
            await this.loadSelectedFile({ keepSettings: true });

            if (this.ui.dataInfo) {
                this.ui.dataInfo.textContent += " (Cropped)";
            }

        } catch (e) {
            console.error("Crop Error:", e);
            alert("Crop Error: " + e.message);
        } finally {
            // Restore UI
            if (this.ui.confirmCropBtn) {
                this.ui.confirmCropBtn.disabled = false;
                this.ui.confirmCropBtn.textContent = "Confirm Crop";
            }
            this.onCropModeExited(); // Force exit UI helper
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

    normalize3DScans() {
        if (!this.originalData || !this.originalMeta) return;

        // Check if data is 3D
        if (this.originalMeta.data_info.ndim !== 3) {
            alert("Normalization is only applicable to 3D datasets.");
            return;
        }

        const [d0, d1, d2] = this.originalMeta.data_info.shape;
        // d0: Energy (rows)
        // d1: Angle (cols)
        // d2: Scan (slices)

        const data = this.originalData;
        const totalIntensities = new Float64Array(d2); // Use Float64 for accumulation precision

        // 1. Calculate Total Intensity for each scan slice
        for (let z = 0; z < d2; z++) {
            let sum = 0;
            for (let y = 0; y < d0; y++) {
                for (let x = 0; x < d1; x++) {
                    const idx = y * d1 * d2 + x * d2 + z;
                    sum += data[idx];
                }
            }
            totalIntensities[z] = sum;
        }

        // 2. Normalize each scan
        let normalizedCount = 0;
        // Recalculate min/max while normalizing
        let newMin = Infinity;
        let newMax = -Infinity;
        const TARGET_INTENSITY = 100.0;

        for (let z = 0; z < d2; z++) {
            const currentSum = totalIntensities[z];

            // Avoid division by zero if a scan is empty/dead
            if (currentSum > 0) {
                const factor = TARGET_INTENSITY / currentSum;

                for (let y = 0; y < d0; y++) {
                    for (let x = 0; x < d1; x++) {
                        const idx = y * d1 * d2 + x * d2 + z;
                        const val = data[idx] * factor;
                        data[idx] = val;

                        // Update stats
                        if (val < newMin) newMin = val;
                        if (val > newMax) newMax = val;
                    }
                }
                normalizedCount++;
            } else {
                // If sum is 0, we can't normalize, just update stats for existing zeros
                for (let y = 0; y < d0; y++) {
                    for (let x = 0; x < d1; x++) {
                        const idx = y * d1 * d2 + x * d2 + z;
                        const val = data[idx];
                        if (val < newMin) newMin = val;
                        if (val > newMax) newMax = val;
                    }
                }
            }
        }

        // 3. Update Metadata Statistics
        // This is CRITICAL for colormap to work correctly, as min/max have changed.
        if (isFinite(newMin) && isFinite(newMax)) {
            this.originalMeta.data_info.min = newMin;
            this.originalMeta.data_info.max = newMax;
        }

        // Update display
        this.recalculateData();

        // Auto-contrast to fit new range
        this.visualizer.autoContrast();
        this.updateContrastUI();

        // Notify user
        if (this.ui.dataInfo) {
            const oldText = this.ui.dataInfo.textContent;
            this.ui.dataInfo.textContent = "Total Intensity Normalization applied (Sum=100).";
            setTimeout(() => {
                if (this.ui.dataInfo) this.ui.dataInfo.textContent = oldText;
            }, 3000);
        }
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
