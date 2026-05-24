# ADAPT - ARPES Data Analysis & Processing Tool

ADAPT is a comprehensive toolkit designed for the visualization and analysis of Angle-Resolved Photoemission Spectroscopy (ARPES) data. It combines a powerful desktop data browser with a high-performance web-based 3D visualization engine.

## Screenshots

| ADAPT Browser | ADAPT Edit |
| :---: | :---: |
| <img src="assets/browser.png" width="100%" alt="ADAPT Browser" /> | <img src="assets/viewer.png" width="100%" alt="ADAPT Edit" /> |

## Features

*   **Data Browser (Desktop)**: A PyQt-based desktop application for browsing, inspecting, and managing ARPES datasets.
*   **3D Viewer (Web)**: An interactive web-based 3D visualizer for exploring volumetric ARPES data (EDC, MDC, isoline cuts).
*   **Integrated Workflow**: Seamlessly switch between the browser and the 3D viewer.
*   **Data Support**: Supports standard ARPES data formats (HDF5/Nexus, Igor Binary Wave, SES ZIP, PXT/PXP).
*   **Brillouin Zone Visualization**: 
    *   3D construction from lattice parameters or Materials Project ID.
    *   Interactive intersection plane visualization (Miller indices, distance, custom color).

## Installation

### 1. Get the Source Code (Two Ways)

**Method A: Download ZIP (Easiest)**
1. Click the blue **"<> Code"** button at the top of this GitHub page.
2. Select **"Download ZIP"**.
3. Extract the ZIP file and open the folder.

**Method B: Git Clone (For Developers)**
```bash
git clone https://github.com/quantum-ctrl/ADAPT.git
cd ADAPT
```

### 2. Environment Setup

Select the instructions for your operating system:

#### macOS
1.  **Install Homebrew** (if not already installed):
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
2.  **Install Python 3**:
    ```bash
    brew install python
    ```
3.  **Create Virtual Environment**:
    ```bash
    python3 -m venv .venv
    ```
4.  **Activate Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```
5.  **Install Dependencies**:
    ```bash
    python -m pip install -r requirements.txt
    ```
    You can also use the helper script:
    ```bash
    ./run.sh install
    ```


#### Windows
1.  **Install Python 3**:
    Download and install Python 3.9+ from [python.org](https://www.python.org/downloads/). Ensure you check **"Add Python to PATH"** during installation.
2.  **Create Virtual Environment**:
    Open PowerShell or Command Prompt and run:
    ```powershell
    python -m venv .venv
    ```
3.  **Activate Virtual Environment**:
    ```powershell
    .\.venv\Scripts\activate
    ```
4.  **Install Dependencies**:
    ```powershell
    python -m pip install -r requirements.txt
    ```


#### Linux (Ubuntu/Debian)
1.  **Install System Dependencies**:
    ```bash
    sudo apt update
    sudo apt install python3 python3-venv python3-pip git
    ```
2.  **Create Virtual Environment**:
    ```bash
    python3 -m venv .venv
    ```
3.  **Activate Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```
4.  **Install Dependencies**:
    ```bash
    python -m pip install -r requirements.txt
    ```
    You can also use the helper script:
    ```bash
    ./run.sh install
    ```


## Usage

### macOS / Linux (using `run.sh`)

The specific components can be launched using the `run.sh` helper script.

If your shell says the script is not executable, run:
```bash
chmod +x run.sh
```

#### Launching Both (Recommended)
```bash
./run.sh both
```

#### Launching Components Individually
- **Data Browser**: `./run.sh browser`
- **ADAPT Edit / 3D Viewer**: `./run.sh edit`

ADAPT Edit is served at `http://127.0.0.1:8000` by default. To use a different
host or port:
```bash
ADAPT_HOST=127.0.0.1 ADAPT_PORT=8001 ./run.sh edit
```

ADAPT Edit must be running before using **Open with Viewer** from the Browser.

### Windows

To run both the Browser and ADAPT Edit, open two terminals from the project
folder after activating `.venv`.

**Terminal 1 (ADAPT Edit / 3D Viewer):**
```powershell
cd ADAPT_edit
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Browser):**
```powershell
cd ADAPT_browser
python app.py
```

## Supported Data

ADAPT can open:

*   HDF5/Nexus files: `.h5`, `.hdf5`, `.nxs`
*   Igor Binary Wave files: `.ibw`
*   SES archives: `.zip`
*   Igor packed experiment files: `.pxt`, `.pxp`

HDF5 loading tries the ADRESS format first and then falls back to SIS.

## Configuration

These environment variables can be set before starting ADAPT Edit:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ADAPT_HOST` | `127.0.0.1` | Host used by the web server. |
| `ADAPT_PORT` | `8000` | Port used by the web server. |
| `ADAPT_DATA_DIR` | `ADAPT_edit/data` | Directory scanned by `/api/files`. |
| `ADAPT_MAX_UPLOAD_SIZE` | `2000000000` | Maximum upload size in bytes. |
| `ADAPT_BROWSE_ROOTS` | data dir and home | Path-list of directories allowed in the web file browser. |
| `ADAPT_ALLOW_FILESYSTEM_BROWSE` | `false` | Set to `1` to allow unrestricted local filesystem browsing. |

On macOS/Linux, separate multiple `ADAPT_BROWSE_ROOTS` entries with `:`. On Windows, use `;`.

## Testing

Run the smoke test suite from the project root:

```bash
python -m unittest discover -s tests
```

Run a Python syntax/import compile check:

```bash
python -m compileall ADAPT_browser ADAPT_edit shared tests
```

## Troubleshooting

*   **Port already in use**: start Edit with another port, for example `ADAPT_PORT=8001 ./run.sh edit`.
*   **Browser opens Edit but the page does not load**: start ADAPT Edit first, then use **Open with Viewer** again.
*   **A directory is blocked in the web file browser**: add it to `ADAPT_BROWSE_ROOTS`, or set `ADAPT_ALLOW_FILESYSTEM_BROWSE=1` for unrestricted local browsing.
*   **Python dependency errors**: activate `.venv` and run `python -m pip install -r requirements.txt`.

## Project Structure

*   `ADAPT_browser/`: Desktop GUI application code.
*   `ADAPT_edit/`: Web server and visualization code.
*   `shared/`: Shared utilities and libraries.
*   `tests/`: Smoke tests for shared loading, saving, and configuration.
*   `requirements.txt`: Python package dependencies.
*   `run.sh`: Main launcher script.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to:

*   Procopios Constantinou for the [ARPEScape](https://github.com/c0deta1ker/ARPEScape) project.
*   Craig Polley for the [pesto](https://gitlab.com/flashingLEDs/pesto) project.
