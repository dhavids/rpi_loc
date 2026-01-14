#!/bin/bash
#
# RPi Localization Auto-completion and Aliases
#
# This script provides bash auto-completion and convenience aliases for
# the rpi_loc image streaming tools.
#
# Source this file in your .bashrc:
#   source /path/to/auto_comp.sh
#

# Configuration - adjust these paths as needed
RPI_LOC_ENV="$HOME/envs/rpi_loc"
RPI_LOC_ROOT="$HOME/rpi_loc"

# Detect repo root if script is sourced from within the repo
if [ -n "${BASH_SOURCE[0]}" ]; then
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    if [ -d "$_SCRIPT_DIR/../../src" ]; then
        RPI_LOC_ROOT="$(cd "$_SCRIPT_DIR/../.." && pwd)"
    fi
fi

# =============================================================================
# Aliases
# =============================================================================

# Clone the rpi_loc repository
alias rpi_loc_clone="cd $HOME && git clone https://github.com/dhavids/rpi_loc.git"

# Update (pull) the rpi_loc repository
alias rpi_loc_update="cd $RPI_LOC_ROOT && git pull"

# Activate the rpi_loc environment
alias rpi_act="source $RPI_LOC_ENV/bin/activate"

# Remove any conflicting alias before defining functions
unalias rpi_loc 2>/dev/null
unalias run_rpi_loc 2>/dev/null

# Activate environment and run the image streamer
rpi_loc() {
    source "$RPI_LOC_ENV/bin/activate"
    python "$RPI_LOC_ROOT/src/image_streamer.py" "$@"
}

# Run image streamer only (assumes environment is already active)
run_rpi_loc() {
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Warning: Virtual environment not active. Run 'rpi_act' first or use 'rpi_loc' instead."
    fi
    python "$RPI_LOC_ROOT/src/image_streamer.py" "$@"
}

# =============================================================================
# Auto-completion for run_rpi_loc and rpi_loc
# =============================================================================

_rpi_loc_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Available options for image_streamer.py
    opts="--help --host --port --resolution --fps --quality --device-id --mock"
    
    case "${prev}" in
        --port)
            # Suggest common ports
            COMPREPLY=( $(compgen -W "5000 5001 5002 8000 8080" -- ${cur}) )
            return 0
            ;;
        --host)
            # Suggest common bind addresses
            COMPREPLY=( $(compgen -W "0.0.0.0 127.0.0.1 localhost" -- ${cur}) )
            return 0
            ;;
        --resolution)
            # Suggest common resolutions
            COMPREPLY=( $(compgen -W "640x480 800x600 1280x720 1920x1080 320x240" -- ${cur}) )
            return 0
            ;;
        --fps)
            # Suggest common frame rates
            COMPREPLY=( $(compgen -W "10 15 24 30 60" -- ${cur}) )
            return 0
            ;;
        --quality)
            # Suggest quality values
            COMPREPLY=( $(compgen -W "50 60 70 80 85 90 95 100" -- ${cur}) )
            return 0
            ;;
        --device-id)
            # Suggest hostname as default device-id
            COMPREPLY=( $(compgen -W "$(hostname)" -- ${cur}) )
            return 0
            ;;
        *)
            ;;
    esac
    
    # If current word starts with -, complete options
    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}

# Register completions for both commands
complete -F _rpi_loc_completions rpi_loc
complete -F _rpi_loc_completions run_rpi_loc

# =============================================================================
# Helper functions
# =============================================================================

# Show rpi_loc help
rpi_loc_help() {
    echo "RPi Localization Commands:"
    echo ""
    echo "  rpi_loc_clone  - Clone the rpi_loc repository from GitHub"
    echo "  rpi_loc_update - Pull latest changes from GitHub"
    echo "  rpi_act        - Activate the rpi_loc virtual environment"
    echo "  rpi_loc        - Activate env and run image streamer (with args)"
    echo "  run_rpi_loc    - Run image streamer only (env must be active)"
    echo ""
    echo "Examples:"
    echo "  rpi_loc_clone                # Clone the repo"
    echo "  rpi_loc_update               # Pull latest changes"
    echo "  rpi_loc --port 5000"
    echo "  rpi_loc --port 5000 --resolution 1280x720 --fps 15"
    echo "  rpi_loc --port 5000 --mock   # Test without camera"
    echo ""
    echo "  rpi_act                      # Just activate environment"
    echo "  run_rpi_loc --port 5000      # Run if env already active"
    echo ""
    echo "Auto-completion available for:"
    echo "  --port, --host, --resolution, --fps, --quality, --device-id, --mock"
    echo ""
}

# Print message when script is sourced
if [ -n "$PS1" ]; then
    echo "RPi Localization commands loaded. Type 'rpi_loc_help' for usage."
fi
