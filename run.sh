#!/bin/bash
# ADAPT - One-click launcher script
# Run both Browser and Viewer applications

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  ADAPT - ARPES Data Analysis & Processing Tool ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo -e "${RED}Error: python3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  browser     Start ADAPT Browser (Desktop GUI)"
    echo "  viewer      Start ADAPT Viewer (Web Server)"
    echo "  both        Start both Browser and Viewer"
    echo "  install     Install dependencies"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 browser   # Start the desktop data browser"
    echo "  $0 viewer    # Start the web visualization server"
    echo "  $0 both      # Start both applications"
}

# Install dependencies
install_deps() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    python3 -m pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
}

# Start Browser
start_browser() {
    echo -e "${GREEN}Starting ADAPT Browser...${NC}"
    cd "$SCRIPT_DIR/ADAPT_browser"
    python3 app.py
}

# Start Viewer
start_viewer() {
    echo -e "${GREEN}Starting ADAPT Viewer on http://localhost:8000${NC}"
    cd "$SCRIPT_DIR/ADAPT_viewer"
    python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
}

# Start both (Viewer in background)
start_both() {
    echo -e "${GREEN}Starting ADAPT Viewer in background...${NC}"
    cd "$SCRIPT_DIR/ADAPT_viewer"
    python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 &
    VIEWER_PID=$!
    echo -e "${BLUE}Viewer PID: $VIEWER_PID${NC}"
    
    sleep 2
    
    echo -e "${GREEN}Starting ADAPT Browser...${NC}"
    cd "$SCRIPT_DIR/ADAPT_browser"
    python3 app.py
    
    # When browser closes, stop viewer
    echo -e "${YELLOW}Browser closed. Stopping Viewer...${NC}"
    kill $VIEWER_PID 2>/dev/null || true
}

# Main
case "${1:-help}" in
    browser)
        start_browser
        ;;
    viewer)
        start_viewer
        ;;
    both)
        start_both
        ;;
    install)
        install_deps
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        usage
        exit 1
        ;;
esac
