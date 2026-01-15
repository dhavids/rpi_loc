#!/usr/bin/env python3
"""
ArUco Marker Generator

Generates printable ArUco markers for TurtleBot localization setup.

Usage:
    python generate_markers.py --output ./markers
    python generate_markers.py --ids 0 1 2 3 10 11 12 --size 200
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def generate_marker(
    marker_id: int,
    size: int = 200,
    dictionary_type: int = cv2.aruco.DICT_4X4_50,
    border_bits: int = 1
) -> np.ndarray:
    """
    Generate a single ArUco marker.
    
    Args:
        marker_id: Marker ID
        size: Size in pixels
        dictionary_type: ArUco dictionary type
        border_bits: Border width in bits
        
    Returns:
        Marker image (grayscale)
    """
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    return marker


def generate_marker_with_label(
    marker_id: int,
    size: int = 200,
    label: str = None,
    dictionary_type: int = cv2.aruco.DICT_4X4_50
) -> np.ndarray:
    """
    Generate a marker with a label below it.
    
    Args:
        marker_id: Marker ID
        size: Marker size in pixels
        label: Optional label (defaults to "ID: {marker_id}")
        dictionary_type: ArUco dictionary type
        
    Returns:
        Marker image with label (grayscale)
    """
    marker = generate_marker(marker_id, size, dictionary_type)
    
    # Create larger image with space for label
    padding = 40
    label_height = 60
    total_height = size + label_height + padding * 2
    total_width = size + padding * 2
    
    # Create white background
    output = np.ones((total_height, total_width), dtype=np.uint8) * 255
    
    # Place marker
    y_start = padding
    x_start = padding
    output[y_start:y_start+size, x_start:x_start+size] = marker
    
    # Add label
    if label is None:
        label = f"ID: {marker_id}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = (total_width - text_size[0]) // 2
    text_y = size + padding + label_height // 2 + text_size[1] // 2
    
    cv2.putText(output, label, (text_x, text_y), font, font_scale, 0, thickness)
    
    return output


def generate_marker_sheet(
    marker_ids: list,
    markers_per_row: int = 4,
    marker_size: int = 150,
    dictionary_type: int = cv2.aruco.DICT_4X4_50
) -> np.ndarray:
    """
    Generate a sheet with multiple markers.
    
    Args:
        marker_ids: List of marker IDs
        markers_per_row: Number of markers per row
        marker_size: Size of each marker
        dictionary_type: ArUco dictionary type
        
    Returns:
        Sheet image with all markers
    """
    # Generate individual markers with labels
    markers = []
    for mid in marker_ids:
        marker = generate_marker_with_label(mid, marker_size, dictionary_type=dictionary_type)
        markers.append(marker)
    
    # Calculate sheet dimensions
    n_markers = len(markers)
    n_rows = (n_markers + markers_per_row - 1) // markers_per_row
    
    marker_h, marker_w = markers[0].shape
    
    sheet_width = markers_per_row * marker_w
    sheet_height = n_rows * marker_h
    
    # Create sheet
    sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
    
    for i, marker in enumerate(markers):
        row = i // markers_per_row
        col = i % markers_per_row
        
        y_start = row * marker_h
        x_start = col * marker_w
        
        sheet[y_start:y_start+marker_h, x_start:x_start+marker_w] = marker
    
    return sheet


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco markers for TurtleBot localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Generate default markers (reference + TurtleBot):
        python generate_markers.py --output ./markers
    
    Generate specific markers:
        python generate_markers.py --ids 0 1 2 3 --output ./markers
    
    Generate a printable sheet:
        python generate_markers.py --sheet --output markers_sheet.png
    
    Custom size:
        python generate_markers.py --ids 10 11 12 --size 300 --output ./markers

Default markers:
    Reference: 0, 1, 2, 3 (corners of coordinate system)
    TurtleBot: 10, 11, 12 (one per robot)
        """
    )
    
    parser.add_argument("--ids", type=int, nargs="+", default=None,
                       help="Marker IDs to generate (default: 0-3, 10-12)")
    parser.add_argument("--size", type=int, default=200,
                       help="Marker size in pixels (default: 200)")
    parser.add_argument("--output", type=str, default="./markers",
                       help="Output directory or file for sheet (default: ./markers)")
    parser.add_argument("--sheet", action="store_true",
                       help="Generate a single sheet with all markers")
    parser.add_argument("--per-row", type=int, default=4,
                       help="Markers per row in sheet mode (default: 4)")
    
    args = parser.parse_args()
    
    # Default marker IDs
    if args.ids is None:
        args.ids = [0, 1, 2, 3, 10, 11, 12]
    
    print(f"Generating markers: {args.ids}")
    print(f"Size: {args.size}px")
    
    if args.sheet:
        # Generate single sheet
        sheet = generate_marker_sheet(
            args.ids,
            markers_per_row=args.per_row,
            marker_size=args.size
        )
        
        output_path = args.output
        if not output_path.endswith(('.png', '.jpg')):
            output_path = args.output + "_sheet.png"
        
        cv2.imwrite(output_path, sheet)
        print(f"Saved marker sheet: {output_path}")
        
    else:
        # Generate individual markers
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for marker_id in args.ids:
            # Determine label based on ID
            if marker_id < 10:
                label = f"REF {marker_id}"
            else:
                label = f"TB3_{marker_id - 10}"
            
            marker = generate_marker_with_label(marker_id, args.size, label)
            
            output_path = output_dir / f"marker_{marker_id:02d}.png"
            cv2.imwrite(str(output_path), marker)
            print(f"  Saved: {output_path}")
        
        print(f"\nGenerated {len(args.ids)} markers in {output_dir}")
    
    print("\nPrinting tips:")
    print("  - Print at 100% scale (no scaling)")
    print("  - Use matte paper to reduce glare")
    print("  - Ensure markers are flat when mounted")
    print("  - Recommended physical size: 5-10cm")


if __name__ == "__main__":
    main()
