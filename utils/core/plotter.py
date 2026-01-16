"""
Position Plotter for TurtleBot Localizer

Real-time matplotlib visualization of TurtleBot positions with trails and velocity vectors.
Based on the Marvelmind plotter design.
"""

import time
import math
from collections import deque
from typing import Dict, Tuple

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from utils.core.setup_logging import get_named_logger

logger = get_named_logger("plotter", __name__)


class PositionPlotter:
    """
    Real-time position plotter with trails and velocity vectors.
    
    Args:
        decay_seconds: Remove points not updated for this long
        trail_seconds: Show trail for this duration
        refresh_interval: Minimum time between redraws
        velocity_scale: Scale factor for velocity arrows
        show_pos: Show position coordinates in labels
    """
    
    def __init__(
        self,
        decay_seconds: float = 15.0,
        trail_seconds: float = 10.0,
        refresh_interval: float = 0.2,
        velocity_scale: float = 1.0,
        show_pos: bool = False,
    ):
        self.decay_seconds = decay_seconds
        self.trail_seconds = trail_seconds
        self.refresh_interval = refresh_interval
        self.velocity_scale = velocity_scale
        self.show_pos = show_pos

        self._points: Dict[Tuple[str, int], deque] = {}
        self._last_draw = 0.0
        self._running = True

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self._setup_axes()
        plt.show(block=False)
        logger.info("Plotter initialized")

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        """Handle key press events."""
        if event.key == "p":
            self.show_pos = not self.show_pos
            logger.info(f"Show positions: {self.show_pos}")
        elif event.key == "c":
            # Clear all trails
            self._points.clear()
            logger.info("Cleared all trails")

    def _setup_axes(self):
        """Setup the plot axes."""
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("TurtleBot Localizer")
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.3)
        
        # Set initial limits (will auto-adjust)
        self.ax.set_xlim(-0.5, 4.5)
        self.ax.set_ylim(-0.5, 4.5)

    def update(self, label: str, beacon_id: int, x: float, y: float, z: float):
        """
        Update position for a tracked object.
        
        Args:
            label: Object type ("BOT" for TurtleBots, "REF" for reference markers)
            beacon_id: Unique ID for the object
            x: X position in world coordinates
            y: Y position in world coordinates
            z: Z position (unused, for compatibility)
        """
        if not self._running:
            return

        now = time.monotonic()
        key = (label, beacon_id)

        if key not in self._points:
            self._points[key] = deque()

        self._points[key].append((x, y, z, now))

        # Remove old points from trail
        while self._points[key] and now - self._points[key][0][3] > self.trail_seconds:
            self._points[key].popleft()

        # Redraw if enough time has passed
        if now - self._last_draw >= self.refresh_interval:
            self._redraw(now)

    def _label_alignment(self, x: float, y: float):
        """Calculate label alignment based on position."""
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        x_mid = (xmin + xmax) * 0.5
        y_mid = (ymin + ymax) * 0.5

        ha = "left" if x < x_mid else "right"
        va = "bottom" if y < y_mid else "top"

        dx = 8 if ha == "left" else -8
        dy = 8 if va == "bottom" else -8

        return ha, va, dx, dy

    def _redraw(self, now: float):
        """Redraw the plot."""
        try:
            self.ax.clear()
            self._setup_axes()

            expired = []

            for (label, bid), samples in self._points.items():
                if not samples:
                    expired.append((label, bid))
                    continue

                last_x, last_y, last_z, last_ts = samples[-1]

                # Check if this object has timed out
                if now - last_ts > self.decay_seconds:
                    expired.append((label, bid))
                    continue

                xs = [p[0] for p in samples]
                ys = [p[1] for p in samples]

                is_bot = label == "BOT"
                color = "red" if is_bot else "blue"
                marker = "o" if is_bot else "s"
                size = 100 if is_bot else 80

                # Draw trail
                if len(xs) > 1 and is_bot:
                    self.ax.plot(xs, ys, color=color, alpha=0.4, linewidth=2)

                # Draw current position
                self.ax.scatter(last_x, last_y, c=color, marker=marker, s=size, zorder=5)

                # Label rendering
                if is_bot:
                    name = f"M{bid}"
                else:
                    name = f"R{bid}"
                
                if self.show_pos:
                    ha, va, dx, dy = self._label_alignment(last_x, last_y)
                    text = f"{name} ({last_x:.2f}, {last_y:.2f})"
                    self.ax.annotate(
                        text,
                        (last_x, last_y),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        ha=ha,
                        va=va,
                        fontsize=9,
                        color=color,
                        alpha=0.9,
                        fontweight="bold"
                    )
                else:
                    self.ax.annotate(
                        name,
                        (last_x, last_y),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=10,
                        color=color,
                        fontweight="bold"
                    )

                # Velocity vector (only for bots)
                if is_bot and len(samples) >= 2:
                    x0, y0, _, t0 = samples[-2]
                    dt = last_ts - t0
                    if dt > 0:
                        vx = (last_x - x0) / dt
                        vy = (last_y - y0) / dt
                        speed = math.hypot(vx, vy)
                        if speed > 0.01:  # Only show if moving
                            self.ax.arrow(
                                last_x,
                                last_y,
                                vx * self.velocity_scale,
                                vy * self.velocity_scale,
                                head_width=0.08,
                                head_length=0.1,
                                fc=color,
                                ec=color,
                                alpha=0.7,
                                zorder=4
                            )

            # Remove expired entries
            for key in expired:
                del self._points[key]

            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            self._last_draw = now

        except KeyboardInterrupt:
            self.close()
        except Exception as e:
            # Don't crash on plotting errors
            logger.debug(f"Plot error: {e}")

    def close(self):
        """Close the plotter."""
        self._running = False
        try:
            plt.close(self.fig)
            logger.info("Plotter closed")
        except Exception:
            pass
