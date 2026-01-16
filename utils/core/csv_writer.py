"""CSV Writer for TurtleBot Position Data

Writes TurtleBot position data to CSV files with throttled rate support.
Based on the Marvelmind CSV writer pattern.
"""

import csv
import time
import logging
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .sink import TurtleBotPosition

logger = logging.getLogger(__name__)


class PositionCSVWriter:
    """
    CSV writer for TurtleBot position data.
    
    Writes position snapshots at a configurable rate with proper
    timestamping for later analysis.
    """
    
    def __init__(
        self,
        output_path: Path,
        rate_hz: float = 10.0,
        append: bool = False
    ):
        """
        Initialize the CSV writer.
        
        Args:
            output_path: Path to output CSV file
            rate_hz: Maximum write rate in Hz
            append: Whether to append to existing file
        """
        self._output_path = Path(output_path)
        self._period = 1.0 / rate_hz
        self._last_write_time = 0.0
        
        # Create parent directories if needed
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file
        mode = "a" if append else "w"
        self._file = self._output_path.open(mode, newline="")
        self._writer = csv.writer(self._file)
        
        # Write header if new file
        if not append or self._output_path.stat().st_size == 0:
            self._writer.writerow([
                "ts_write",        # Wall clock time when written
                "ts_frame",        # Frame timestamp
                "frame_number",    # Frame number
                "local_id",        # Local tracking ID
                "name",            # TurtleBot name
                "x",               # World X coordinate
                "y",               # World Y coordinate
                "theta",           # Orientation (radians)
                "confidence",      # Detection confidence
                "pixel_x",         # Image X coordinate
                "pixel_y",         # Image Y coordinate
                "homography_valid" # Whether world coords are valid
            ])
            self._file.flush()
        
        # Statistics
        self._rows_written = 0
        self._snapshots_written = 0
        
        logger.info(
            f"CSV writer opened at {output_path} (rate {rate_hz:.1f} Hz)"
        )
    
    def write_snapshot(
        self,
        turtlebots: List["TurtleBotPosition"],
        frame_timestamp: float,
        frame_number: int,
        homography_valid: bool
    ) -> bool:
        """
        Write a position snapshot.
        
        Args:
            turtlebots: List of TurtleBot positions
            frame_timestamp: Timestamp of the frame
            frame_number: Frame number
            homography_valid: Whether world coordinates are valid
            
        Returns:
            True if written, False if throttled
        """
        now = time.monotonic()
        
        # Rate limiting
        if now - self._last_write_time < self._period:
            return False
        
        ts_write = time.time()
        
        rows = []
        for tb in turtlebots:
            rows.append([
                f"{ts_write:.6f}",
                f"{frame_timestamp:.6f}",
                frame_number,
                tb.local_id,
                tb.name,
                f"{tb.x:.6f}",
                f"{tb.y:.6f}",
                f"{tb.theta:.6f}",
                f"{tb.confidence:.4f}",
                tb.pixel_x,
                tb.pixel_y,
                1 if homography_valid else 0
            ])
        
        if not rows:
            return False
        
        self._writer.writerows(rows)
        self._file.flush()
        
        self._last_write_time = now
        self._rows_written += len(rows)
        self._snapshots_written += 1
        
        logger.debug(
            f"CSV snapshot written: {len(rows)} TurtleBots at frame {frame_number}"
        )
        
        return True
    
    def close(self):
        """Close the CSV file."""
        logger.info(
            f"Closing CSV writer. Wrote {self._rows_written} rows in "
            f"{self._snapshots_written} snapshots"
        )
        self._file.close()
    
    @property
    def rows_written(self) -> int:
        """Get total rows written."""
        return self._rows_written
    
    @property
    def snapshots_written(self) -> int:
        """Get total snapshots written."""
        return self._snapshots_written
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
