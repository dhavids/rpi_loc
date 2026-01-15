#!/usr/bin/env python3
"""
TurtleBot Localizer

Combines image streaming and ArUco-based localization to track TurtleBots
in real-time from a Raspberry Pi camera stream.

Usage:
    python localizer.py --host 100.99.98.1 --port 5000
    python localizer.py --config markers.yaml

The localizer:
1. Connects to the RPi camera stream
2. Detects ArUco markers in each frame
3. Uses reference markers to establish world coordinates
4. Tracks TurtleBot positions and orientations
5. Optionally publishes poses via callback or saves to file
"""

import argparse
import json
import logging
import signal
import sys
import time
from typing import Dict, List, Optional
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_listener import ImageListener
from utils.img_proc.image_parser import (
    ImageParser,
    MarkerConfig,
    TurtleBotConfig,
    TurtleBotPose,
    LocalizationResult,
    create_default_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TurtleBotLocalizer:
    """
    Real-time TurtleBot localization from camera stream.
    
    Combines ImageListener and ImageParser to provide continuous
    localization of TurtleBots using ArUco markers.
    """
    
    def __init__(
        self,
        host: str = "100.99.98.1",
        port: int = 5000,
        reference_markers: Optional[List[MarkerConfig]] = None,
        turtlebot_markers: Optional[List[TurtleBotConfig]] = None,
        display: bool = True,
        save_poses: Optional[str] = None,
        on_pose_update: Optional[callable] = None
    ):
        """
        Initialize the localizer.
        
        Args:
            host: RPi streamer host address
            port: RPi streamer port
            reference_markers: Reference marker configurations (or use defaults)
            turtlebot_markers: TurtleBot marker configurations (or use defaults)
            display: Whether to display annotated video
            save_poses: Path to save pose history (JSON)
            on_pose_update: Callback function(poses: List[TurtleBotPose]) for each frame
        """
        # Use default config if not provided
        if reference_markers is None or turtlebot_markers is None:
            ref_markers, tb_markers = create_default_config()
            reference_markers = reference_markers or ref_markers
            turtlebot_markers = turtlebot_markers or tb_markers
        
        # Initialize parser
        self.parser = ImageParser(reference_markers, turtlebot_markers)
        
        # Initialize listener with frame callback
        self.listener = ImageListener(
            host=host,
            port=port,
            display=False,  # We'll handle display ourselves
            on_frame=self._on_frame
        )
        
        self.display = display
        self.save_poses = save_poses
        self.on_pose_update = on_pose_update
        
        # State
        self._running = False
        self._pose_history: List[Dict] = []
        self._current_poses: Dict[str, TurtleBotPose] = {}
        self._last_result: Optional[LocalizationResult] = None
        
        # Display window name
        self._window_name = "TurtleBot Localizer"
        
        logger.info(f"Localizer initialized - connecting to {host}:{port}")
    
    def _on_frame(self, frame: np.ndarray, metadata: Dict):
        """
        Callback for each received frame.
        
        Args:
            frame: BGR image
            metadata: Frame metadata from streamer
        """
        timestamp = metadata.get('timestamp_ms', time.time() * 1000) / 1000.0
        frame_number = metadata.get('frame_number', 0)
        
        # Process frame for localization
        result = self.parser.process_frame(frame, timestamp, frame_number)
        self._last_result = result
        
        # Update current poses
        for pose in result.turtlebots:
            self._current_poses[pose.name] = pose
        
        # Save pose history if enabled
        if self.save_poses:
            self._record_poses(result)
        
        # Call user callback if provided
        if self.on_pose_update and result.turtlebots:
            self.on_pose_update(result.turtlebots)
        
        # Display annotated frame
        if self.display:
            annotated = self.parser.draw_detections(frame, result)
            self._draw_extra_info(annotated, metadata)
            cv2.imshow(self._window_name, annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._running = False
            elif key == ord('c'):
                # Calibration mode - print detected marker positions
                self._print_calibration_info()
            elif key == ord('s'):
                # Save current frame
                self._save_snapshot(annotated, frame_number)
    
    def _draw_extra_info(self, frame: np.ndarray, metadata: Dict):
        """Draw additional info on frame."""
        # Draw stream info in top-right
        h, w = frame.shape[:2]
        info_lines = [
            f"Device: {metadata.get('device_id', '?')}",
            f"Frame: {metadata.get('frame_number', '?')}",
            f"Stream FPS: {self.listener.fps:.1f}"
        ]
        
        y = 25
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x = w - text_size[0] - 10
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += 20
        
        # Draw TurtleBot pose summary
        y = 25
        for name, pose in self._current_poses.items():
            if self._last_result and self._last_result.homography_valid:
                text = f"{name}: x={pose.x:.3f} y={pose.y:.3f} θ={np.degrees(pose.theta):.1f}°"
            else:
                text = f"{name}: px=({pose.pixel_position[0]}, {pose.pixel_position[1]})"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += 20
    
    def _record_poses(self, result: LocalizationResult):
        """Record poses to history."""
        record = {
            "timestamp": result.timestamp,
            "frame_number": result.frame_number,
            "homography_valid": result.homography_valid,
            "poses": [
                {
                    "name": p.name,
                    "x": p.x,
                    "y": p.y,
                    "theta": p.theta,
                    "confidence": p.confidence
                }
                for p in result.turtlebots
            ]
        }
        self._pose_history.append(record)
    
    def _print_calibration_info(self):
        """Print calibration information for setting up markers."""
        logger.info("=== Calibration Info ===")
        logger.info("Detected reference markers:")
        
        # Re-detect to get current marker positions
        # This would need access to the current frame
        logger.info("Press 'c' while viewing a frame with all markers visible")
        logger.info("to capture their pixel positions for calibration.")
    
    def _save_snapshot(self, frame: np.ndarray, frame_number: int):
        """Save current frame as snapshot."""
        filename = f"snapshot_{frame_number:06d}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Saved snapshot: {filename}")
    
    def start(self, auto_reconnect: bool = True):
        """
        Start the localizer.
        
        Args:
            auto_reconnect: Whether to auto-reconnect on connection loss
        """
        self._running = True
        
        logger.info("Starting TurtleBot Localizer...")
        logger.info("Controls: q=quit, c=calibration info, s=save snapshot")
        
        try:
            self.listener.start(auto_reconnect=auto_reconnect)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the localizer and save data."""
        self._running = False
        self.listener.stop()
        
        if self.display:
            cv2.destroyAllWindows()
        
        # Save pose history
        if self.save_poses and self._pose_history:
            with open(self.save_poses, 'w') as f:
                json.dump(self._pose_history, f, indent=2)
            logger.info(f"Saved {len(self._pose_history)} pose records to {self.save_poses}")
        
        # Print statistics
        stats = self.parser.stats
        logger.info(f"Localizer stopped. Stats: {stats}")
    
    def get_poses(self) -> Dict[str, TurtleBotPose]:
        """Get current TurtleBot poses."""
        return self._current_poses.copy()
    
    def get_pose(self, name: str) -> Optional[TurtleBotPose]:
        """Get pose for a specific TurtleBot."""
        return self._current_poses.get(name)


def load_config(config_path: str) -> tuple:
    """
    Load marker configuration from JSON file.
    
    Config format:
    {
        "reference_markers": [
            {"id": 0, "x": 0.0, "y": 0.0},
            {"id": 1, "x": 2.0, "y": 0.0},
            ...
        ],
        "turtlebot_markers": [
            {"id": 10, "name": "tb3_0"},
            {"id": 11, "name": "tb3_1"},
            ...
        ]
    }
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    ref_markers = [
        MarkerConfig(marker_id=m['id'], world_position=(m['x'], m['y']))
        for m in config.get('reference_markers', [])
    ]
    
    tb_markers = [
        TurtleBotConfig(marker_id=m['id'], name=m['name'])
        for m in config.get('turtlebot_markers', [])
    ]
    
    return ref_markers, tb_markers


def main():
    parser = argparse.ArgumentParser(
        description="TurtleBot Localizer - Real-time localization from camera stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Basic usage:
        python localizer.py
    
    With custom host:
        python localizer.py --host 100.99.98.1 --port 5000
    
    Save pose history:
        python localizer.py --save-poses poses.json
    
    Load custom marker config:
        python localizer.py --config markers.json
    
    Headless mode:
        python localizer.py --no-display --save-poses poses.json

Controls (when display is enabled):
    q - Quit
    c - Print calibration info
    s - Save current frame as snapshot
        """
    )
    
    parser.add_argument("--host", default="100.99.98.1",
                       help="Streamer host address (default: 100.99.98.1)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Streamer port number (default: 5000)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to marker configuration JSON file")
    parser.add_argument("--save-poses", type=str, default=None,
                       help="Path to save pose history (JSON)")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable display window")
    parser.add_argument("--no-reconnect", action="store_true",
                       help="Disable auto-reconnect")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    ref_markers, tb_markers = None, None
    if args.config:
        ref_markers, tb_markers = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    
    # Create and start localizer
    localizer = TurtleBotLocalizer(
        host=args.host,
        port=args.port,
        reference_markers=ref_markers,
        turtlebot_markers=tb_markers,
        display=not args.no_display,
        save_poses=args.save_poses
    )
    
    localizer.start(auto_reconnect=not args.no_reconnect)


if __name__ == "__main__":
    main()
