# ADAPT - ARPES Data Analysis & Processing Tool

ADAPT is a comprehensive toolkit designed for the visualization and analysis of Angle-Resolved Photoemission Spectroscopy (ARPES) data. It combines a powerful desktop data browser with a high-performance web-based 3D visualization engine.

## Features

*   **Data Browser (Desktop)**: A PyQt-based desktop application for browsing, inspecting, and managing ARPES datasets.
*   **3D Viewer (Web)**: An interactive web-based 3D visualizer for exploring volumetric ARPES data (EDC, MDC, isoline cuts).
*   **Integrated Workflow**: Seamlessly switch between the browser and the 3D viewer.
*   **Data Support**: Supports standard ARPES data formats (HDF5, Igor Binary Wave).

## Installation

### Prerequisites

*   Python 3.9 or higher
*   Git

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/quantum-ctrl/adapt.git
    cd ADAPT
    ```

2.  Install dependencies using the provided script:
    ```bash
    ./run.sh install
    ```
    Or manually via pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The specific components can be launched using the `run.sh` helper script.

### Launching the Data Browser
To start the desktop GUI for browsing files:
```bash
./run.sh browser
```

### Launching the 3D Viewer
To start the web-based visualization server (default: http://localhost:8000):
```bash
./run.sh viewer
```

### Launching Both
To run both the browser and viewer simultaneously (recommended):
```bash
./run.sh both
```

## Project Structure

*   `ADAPT_browser/`: Desktop GUI application code.
*   `ADAPT_viewer/`: Web server and visualization code.
*   `shared/`: Shared utilities and libraries.
*   `requirements.txt`: Python package dependencies.
*   `run.sh`: Main launcher script.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to:

*   Procopios Constantinou for the [ARPEScape](https://github.com/c0deta1ker/ARPEScape) project.
*   Craig Polley for the [pesto](https://gitlab.com/flashingLEDs/pesto) project.

