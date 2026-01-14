#!/bin/bash
#
# RPi Localization Setup Script
# 
# This script sets up the Python environment and dependencies for the rpi_loc
# repository on a Raspberry Pi running Raspberry Pi OS.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# What this script does:
#   1. Updates apt packages
#   2. Installs python3-venv if not present
#   3. Creates a Python virtual environment at ~/envs/rpi_loc
#   4. Installs all required Python dependencies
#   5. Activates the environment
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_DIR="$HOME/envs/rpi_loc"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    RPi Localization Setup Script${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Check if running on Raspberry Pi
check_rpi() {
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(cat /proc/device-tree/model)
        if [[ "$MODEL" == *"Raspberry Pi"* ]]; then
            print_status "Detected: $MODEL"
            return 0
        fi
    fi
    print_warning "Not running on a Raspberry Pi (or cannot detect model)"
    print_info "Continuing anyway..."
    return 0
}

# Update apt packages
update_apt() {
    print_info "Updating apt packages..."
    sudo apt update -y
    print_status "Apt packages updated"
}

# Install python3-venv if needed
install_venv_package() {
    if ! dpkg -l | grep -q python3-venv; then
        print_info "Installing python3-venv..."
        sudo apt install -y python3-venv python3-pip
        print_status "python3-venv installed"
    else
        print_status "python3-venv already installed"
    fi
}

# Install system dependencies for picamera2 and opencv
install_system_deps() {
    print_info "Installing system dependencies..."
    
    # Install picamera2 (Raspberry Pi camera library)
    if ! dpkg -l | grep -q python3-picamera2; then
        print_info "Installing picamera2..."
        sudo apt install -y python3-picamera2
        print_status "picamera2 installed"
    else
        print_status "picamera2 already installed"
    fi
    
    # Install OpenCV dependencies
    print_info "Installing OpenCV system dependencies..."
    sudo apt install -y \
        libopencv-dev \
        python3-opencv \
        libatlas-base-dev \
        libjasper-dev \
        libqtgui4 \
        libqt4-test \
        libhdf5-dev \
        2>/dev/null || true  # Some packages may not be available on all RPi OS versions
    
    print_status "System dependencies installed"
}

# Create virtual environment
create_venv() {
    if [ -d "$ENV_DIR" ]; then
        print_status "Virtual environment already exists at $ENV_DIR"
        return 0
    fi
    
    print_info "Creating virtual environment at $ENV_DIR..."
    
    # Create envs directory if it doesn't exist
    mkdir -p "$HOME/envs"
    
    # Create virtual environment with system site packages
    # This allows access to system-installed packages like picamera2
    python3 -m venv "$ENV_DIR" --system-site-packages
    
    print_status "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source "$ENV_DIR/bin/activate"
    print_status "Virtual environment activated"
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Core dependencies for image streaming
    pip install \
        numpy \
        opencv-python-headless \
        netifaces
    
    # Optional: Install development dependencies
    pip install \
        pytest \
        black \
        flake8
    
    print_status "Python dependencies installed"
}

# Create requirements.txt if it doesn't exist
create_requirements() {
    REQUIREMENTS_FILE="$REPO_ROOT/requirements.txt"
    
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_info "Creating requirements.txt..."
        cat > "$REQUIREMENTS_FILE" << EOF
# RPi Localization Dependencies
# Install with: pip install -r requirements.txt

# Core dependencies
numpy>=1.20.0
opencv-python-headless>=4.5.0
netifaces>=0.11.0

# Note: picamera2 should be installed via apt on Raspberry Pi:
#   sudo apt install python3-picamera2

# Development dependencies (optional)
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
EOF
        print_status "requirements.txt created"
    else
        print_status "requirements.txt already exists"
    fi
}

# Print activation instructions
print_instructions() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}    Setup Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "To activate the environment in the future, run:"
    echo -e "  ${YELLOW}source ~/envs/rpi_loc/bin/activate${NC}"
    echo ""
    echo -e "Or add this alias to your ~/.bashrc:"
    echo -e "  ${YELLOW}alias rpi_loc='source ~/envs/rpi_loc/bin/activate'${NC}"
    echo ""
    echo -e "To run the image streamer:"
    echo -e "  ${YELLOW}cd $REPO_ROOT${NC}"
    echo -e "  ${YELLOW}python src/image_streamer.py --port 5000${NC}"
    echo ""
    echo -e "For testing without a camera:"
    echo -e "  ${YELLOW}python src/image_streamer.py --port 5000 --mock${NC}"
    echo ""
}

# Add convenience alias to bashrc
add_alias() {
    ALIAS_LINE="alias rpi_loc='source ~/envs/rpi_loc/bin/activate && cd $REPO_ROOT'"
    
    if ! grep -q "alias rpi_loc=" "$HOME/.bashrc" 2>/dev/null; then
        print_info "Adding convenience alias to ~/.bashrc..."
        echo "" >> "$HOME/.bashrc"
        echo "# RPi Localization environment alias" >> "$HOME/.bashrc"
        echo "$ALIAS_LINE" >> "$HOME/.bashrc"
        print_status "Alias 'rpi_loc' added to ~/.bashrc"
    else
        print_status "Alias already exists in ~/.bashrc"
    fi
}

# Main execution
main() {
    check_rpi
    echo ""
    
    update_apt
    echo ""
    
    install_venv_package
    echo ""
    
    install_system_deps
    echo ""
    
    create_venv
    echo ""
    
    activate_venv
    echo ""
    
    install_python_deps
    echo ""
    
    create_requirements
    echo ""
    
    add_alias
    
    print_instructions
    
    # Keep the environment activated for the current shell
    echo -e "${GREEN}Environment is now active in this shell.${NC}"
}

# Run main function
main
