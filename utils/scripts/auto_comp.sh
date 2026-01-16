#!/bin/bash
#
# RPi Localization Auto-completion and Aliases
#
# This script provides bash auto-completion and convenience aliases for
# the rpi_loc tools. It detects whether it's running on an RPi or a
# development machine and loads appropriate commands.
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
# Platform Detection
# =============================================================================

_is_raspberry_pi() {
    # Check multiple indicators for RPi
    if [ -f /proc/device-tree/model ]; then
        grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null && return 0
    fi
    if [ -f /etc/rpi-issue ]; then
        return 0
    fi
    if command -v vcgencmd &>/dev/null; then
        return 0
    fi
    return 1
}

IS_RPI=false
if _is_raspberry_pi; then
    IS_RPI=true
fi

# =============================================================================
# Common Aliases (all platforms)
# =============================================================================

# Clone the rpi_loc repository
alias rpi_loc_clone="cd $HOME && git clone https://github.com/dhavids/rpi_loc.git"

# Update (pull) the rpi_loc repository
alias rpi_loc_update="cd $RPI_LOC_ROOT && git pull"

# Activate the rpi_loc environment
alias rpi_act="source $RPI_LOC_ENV/bin/activate"

# =============================================================================
# RPi-specific commands (image streaming)
# =============================================================================

if [ "$IS_RPI" = true ]; then
    # Remove any conflicting alias before defining functions
    unalias rpi_loc 2>/dev/null
    unalias run_rpi_loc 2>/dev/null

    # Activate environment and run the image streamer
    rpi_loc() {
        source "$RPI_LOC_ENV/bin/activate"
        python "$RPI_LOC_ROOT/src/image_streamer.py --binned" "$@"
    }

    # Run image streamer only (assumes environment is already active)
    run_rpi_loc() {
        if [ -z "$VIRTUAL_ENV" ]; then
            echo "Warning: Virtual environment not active. Run 'rpi_act' first or use 'rpi_loc' instead."
        fi
        python "$RPI_LOC_ROOT/src/image_streamer.py" "$@"
    }
fi

# =============================================================================
# Development machine commands (model training CLIs)
# =============================================================================

if [ "$IS_RPI" = false ]; then
    # Remove any conflicting aliases
    unalias yolo_loc 2>/dev/null
    unalias ssd_loc 2>/dev/null
    unalias tb_loc 2>/dev/null

    # YOLO TurtleBot detection CLI
    # Must run from parent of rpi_loc for module imports to work
    yolo_loc() {
        if [ -z "$VIRTUAL_ENV" ]; then
            source "$RPI_LOC_ENV/bin/activate"
        fi
        local _orig_dir="$PWD"
        cd "$(dirname "$RPI_LOC_ROOT")" || return 1
        python -m rpi_loc.src.models.yolo.cli "$@"
        local _exit_code=$?
        cd "$_orig_dir"
        return $_exit_code
    }

    # SSD TurtleBot detection CLI (placeholder)
    # Must run from parent of rpi_loc for module imports to work
    ssd_loc() {
        if [ -z "$VIRTUAL_ENV" ]; then
            source "$RPI_LOC_ENV/bin/activate"
        fi
        local _orig_dir="$PWD"
        cd "$(dirname "$RPI_LOC_ROOT")" || return 1
        python -m rpi_loc.src.models.ssd.cli "$@"
        local _exit_code=$?
        cd "$_orig_dir"
        return $_exit_code
    }

    # TurtleBot Localizer - real-time localization from camera stream
    # Must run from parent of rpi_loc for module imports to work
    tb_loc() {
        if [ -z "$VIRTUAL_ENV" ]; then
            source "$RPI_LOC_ENV/bin/activate"
        fi
        local _orig_dir="$PWD"
        cd "$(dirname "$RPI_LOC_ROOT")" || return 1
        python -m rpi_loc.src.localizer "$@"
        local _exit_code=$?
        cd "$_orig_dir"
        return $_exit_code
    }
fi

# =============================================================================
# Auto-completion for RPi commands
# =============================================================================

if [ "$IS_RPI" = true ]; then
    _rpi_loc_completions() {
        local cur prev opts
        COMPREPLY=()
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        
        # Available options for image_streamer.py
        opts="--help --host --port --resolution --fps --quality --device-id --mock"
        
        case "${prev}" in
            --port)
                COMPREPLY=( $(compgen -W "5000 5001 5002 8000 8080" -- ${cur}) )
                return 0
                ;;
            --host)
                COMPREPLY=( $(compgen -W "0.0.0.0 127.0.0.1 localhost" -- ${cur}) )
                return 0
                ;;
            --resolution)
                COMPREPLY=( $(compgen -W "640x480 800x600 1280x720 1920x1080 320x240" -- ${cur}) )
                return 0
                ;;
            --fps)
                COMPREPLY=( $(compgen -W "10 15 24 30 60" -- ${cur}) )
                return 0
                ;;
            --quality)
                COMPREPLY=( $(compgen -W "50 60 70 80 85 90 95 100" -- ${cur}) )
                return 0
                ;;
            --device-id)
                COMPREPLY=( $(compgen -W "$(hostname)" -- ${cur}) )
                return 0
                ;;
            *)
                ;;
        esac
        
        if [[ ${cur} == -* ]]; then
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
        fi
    }

    complete -F _rpi_loc_completions rpi_loc
    complete -F _rpi_loc_completions run_rpi_loc
fi

# =============================================================================
# Auto-completion for YOLO CLI
# =============================================================================

if [ "$IS_RPI" = false ]; then
    _yolo_loc_completions() {
        local cur prev opts commands
        COMPREPLY=()
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        
        # Main commands
        commands="collect annotate create-dataset train finetune evaluate export predict"
        
        # Get the subcommand (first non-option argument)
        local cmd=""
        for ((i=1; i < COMP_CWORD; i++)); do
            if [[ "${COMP_WORDS[i]}" != -* ]]; then
                cmd="${COMP_WORDS[i]}"
                break
            fi
        done
        
        case "${cmd}" in
            collect)
                opts="--camera --video --stream --host --port --duration --preview --output --interval --max-images --every-n"
                ;;
            annotate)
                opts="--images --labels --classes --progress"
                ;;
            create-dataset)
                opts="--images --labels --output --classes --train-ratio --val-ratio --test-ratio --no-shuffle --seed"
                ;;
            train)
                opts="--data --base-model --epochs --batch-size --image-size --device --name --resume --workers"
                ;;
            finetune)
                opts="--data --base-model --epochs --batch-size --freeze-layers --workers"
                ;;
            evaluate)
                opts="--model --data"
                ;;
            export)
                opts="--model --format --output"
                ;;
            predict)
                opts="--model --source --confidence --no-save"
                ;;
            *)
                # No subcommand yet, complete with commands
                COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
                return 0
                ;;
        esac
        
        # Complete options
        if [[ ${cur} == -* ]]; then
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
        fi
        
        # Complete file/directory paths for path arguments
        case "${prev}" in
            --output|-o|--images|-i|--labels|-l|--data|-d|--model|-m|--source|-s|--video)
                COMPREPLY=( $(compgen -f -- ${cur}) )
                return 0
                ;;
            --format|-f)
                COMPREPLY=( $(compgen -W "onnx torchscript openvino coreml tflite" -- ${cur}) )
                return 0
                ;;
            --device)
                COMPREPLY=( $(compgen -W "auto cpu cuda mps" -- ${cur}) )
                return 0
                ;;
            --base-model)
                COMPREPLY=( $(compgen -W "yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt" -- ${cur}) )
                return 0
                ;;
        esac
    }

    _tb_loc_completions() {
        local cur prev opts
        COMPREPLY=()
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        
        opts="--host --port --config --model --confidence --broadcast --broadcast-port --csv --no-display --no-reconnect --timeout --help"
        
        case "${prev}" in
            --host)
                COMPREPLY=( $(compgen -W "100.99.98.1 localhost 127.0.0.1" -- ${cur}) )
                return 0
                ;;
            --port|--broadcast-port)
                COMPREPLY=( $(compgen -W "5000 5555 5001" -- ${cur}) )
                return 0
                ;;
            --config|--model|-m|--csv)
                COMPREPLY=( $(compgen -f -- ${cur}) )
                return 0
                ;;
            --confidence|--timeout)
                return 0
                ;;
        esac
        
        if [[ ${cur} == -* ]]; then
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
        fi
    }

    complete -F _yolo_loc_completions yolo_loc
    complete -F _tb_loc_completions tb_loc
fi

# =============================================================================
# Helper functions
# =============================================================================

# Show rpi_loc help
rpi_loc_help() {
    echo "RPi Localization Commands:"
    echo ""
    echo "Common (all platforms):"
    echo "  rpi_loc_clone  - Clone the rpi_loc repository from GitHub"
    echo "  rpi_loc_update - Pull latest changes from GitHub"
    echo "  rpi_act        - Activate the rpi_loc virtual environment"
    echo ""
    
    if [ "$IS_RPI" = true ]; then
        echo "RPi commands (image streaming):"
        echo "  rpi_loc        - Activate env and run image streamer (with args)"
        echo "  run_rpi_loc    - Run image streamer only (env must be active)"
        echo ""
        echo "Examples:"
        echo "  rpi_loc --port 5000"
        echo "  rpi_loc --port 5000 --resolution 1280x720 --fps 15"
        echo "  rpi_loc --port 5000 --mock   # Test without camera"
        echo ""
    else
        echo "Development commands:"
        echo "  tb_loc         - TurtleBot localizer (real-time tracking)"
        echo "  yolo_loc       - YOLO TurtleBot detection training CLI"
        echo "  ssd_loc        - SSD TurtleBot detection training CLI"
        echo ""
        echo "Localizer examples:"
        echo "  tb_loc --host 100.99.98.1              # Basic (display only)"
        echo "  tb_loc --host 100.99.98.1 --broadcast  # With position broadcasting"
        echo "  tb_loc --host 100.99.98.1 --csv pos.csv  # With CSV logging"
        echo "  tb_loc --host 100.99.98.1 --broadcast --csv pos.csv  # Full setup"
        echo "  tb_loc --model path/to/best.pt        # Custom YOLO model"
        echo ""
        echo "YOLO CLI examples:"
        echo "  yolo_loc collect --stream              # Collect images from RPi stream"
        echo "  yolo_loc annotate                      # Annotate collected images"
        echo "  yolo_loc create-dataset                # Create train/val/test split"
        echo "  yolo_loc train --epochs 100            # Train model"
        echo "  yolo_loc evaluate --model best.pt     # Evaluate model"
        echo "  yolo_loc predict --model best.pt -s img.jpg  # Run inference"
        echo ""
        echo "Default paths (in files/models/yolo/):"
        echo "  Images:  files/models/yolo/data/images"
        echo "  Labels:  files/models/yolo/data/labels"
        echo "  Dataset: files/models/yolo/data/dataset"
        echo "  Model:   files/models/yolo/runs/detect/turtlebot/weights/best.pt"
        echo ""
    fi
    
    echo "Auto-completion available for all commands."
    echo ""
}

# Print message when script is sourced
if [ -n "$PS1" ]; then
    if [ "$IS_RPI" = true ]; then
        echo "RPi Localization (RPi mode) loaded. Type 'rpi_loc_help' for usage."
    else
        echo "RPi Localization (Dev mode) loaded. Type 'rpi_loc_help' for usage."
    fi
fi
