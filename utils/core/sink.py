"""Position Sink

Unified output handler for position data.
Publishes to both CSV writer and broadcaster if configured.
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .csv_writer import PositionCSVWriter
    from .broadcaster import PositionBroadcaster

logger = logging.getLogger(__name__)


@dataclass
class TurtleBotPosition:
    """Position data for a single TurtleBot."""
    name: str
    local_id: int
    x: float
    y: float
    theta: float
    confidence: float
    pixel_x: int
    pixel_y: int
    timestamp: float


@dataclass
class ReferenceMarkerPosition:
    """Position data for a reference marker."""
    name: str           # R0, R1, R2, R3
    marker_id: int      # ArUco marker ID
    world_x: float      # Fixed world X coordinate
    world_y: float      # Fixed world Y coordinate
    pixel_x: int        # Current pixel X coordinate (from detection)
    pixel_y: int        # Current pixel Y coordinate (from detection)
    detected: bool      # Whether marker was detected in this frame


class PositionSink:
    """
    Unified sink for position data.
    
    Publishes to both CSV writer and broadcaster if configured.
    """
    
    def __init__(
        self,
        csv_writer: Optional["PositionCSVWriter"] = None,
        broadcaster: Optional["PositionBroadcaster"] = None
    ):
        """
        Initialize the position sink.
        
        Args:
            csv_writer: Optional CSV writer
            broadcaster: Optional broadcaster
        """
        self.csv_writer = csv_writer
        self.broadcaster = broadcaster
        
        self._paused = False
    
    def publish(
        self,
        turtlebots: List["TurtleBotPosition"],
        reference_markers: List["ReferenceMarkerPosition"],
        frame_timestamp: float,
        frame_number: int,
        homography_valid: bool
    ):
        """
        Publish position data to all configured outputs.
        
        Args:
            turtlebots: List of TurtleBot positions
            reference_markers: List of reference marker positions
            frame_timestamp: Frame timestamp
            frame_number: Frame number
            homography_valid: Whether homography is valid
        """
        if self._paused:
            return
        
        ts_pub = time.time()
        
        # Write to CSV (turtlebots only for now)
        if self.csv_writer:
            self.csv_writer.write_snapshot(
                turtlebots,
                frame_timestamp,
                frame_number,
                homography_valid
            )
        
        # Broadcast (includes both reference markers and turtlebots)
        if self.broadcaster:
            # Reference markers: R0=(0,0), R1=(4,0), R2=(0,4), R3=(4,4)
            ref_markers_payload = {}
            for rm in reference_markers:
                ref_markers_payload[rm.name] = {
                    "marker_id": rm.marker_id,
                    "x": rm.world_x,
                    "y": rm.world_y,
                    "pixel": {"x": rm.pixel_x, "y": rm.pixel_y},
                    "detected": rm.detected
                }
            
            # TurtleBots: M0, M1, M2, ... (using local_id as index)
            turtlebots_payload = {}
            for tb in turtlebots:
                tb_name = f"M{tb.local_id}"
                turtlebots_payload[tb_name] = {
                    "local_id": tb.local_id,
                    "x": tb.x,
                    "y": tb.y,
                    "theta": tb.theta,
                    "confidence": tb.confidence,
                    "pixel": {"x": tb.pixel_x, "y": tb.pixel_y}
                }
            
            payload = {
                "ts_pub": ts_pub,
                "frame_number": frame_number,
                "frame_timestamp": frame_timestamp,
                "homography_valid": homography_valid,
                "reference_markers": ref_markers_payload,
                "turtlebots": turtlebots_payload
            }
            self.broadcaster.update(payload)
    
    def pause(self):
        """Pause publishing."""
        self._paused = True
        if self.broadcaster:
            self.broadcaster.pause()
    
    def resume(self):
        """Resume publishing."""
        self._paused = False
    
    def close(self):
        """Close all outputs."""
        if self.csv_writer:
            self.csv_writer.close()
        if self.broadcaster:
            self.broadcaster.stop()
