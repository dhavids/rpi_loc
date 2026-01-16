#!/usr/bin/env python3
"""
ArUco Marker Generator

Generates printable ArUco markers for Object detection based localization setup.
Outputs A4-sized pages by default.

Usage:
    python generate_markers.py --output ./markers
    python generate_markers.py --ids 0 1 2 3 10 11 12 --size 200
"""

import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# A4 paper size in pixels at 300 DPI
A4_WIDTH_PX = 2480   # 210mm at 300 DPI
A4_HEIGHT_PX = 3508  # 297mm at 300 DPI

# Default marker definitions
# Reference markers (corners of coordinate system)
REFERENCE_MARKERS = {
    0: "TL",   # Top Left
    1: "TR",   # Top Right
    2: "BL",   # Bottom Left
    3: "BR",   # Bottom Right
}

# TurtleBot markers
TURTLEBOT_MARKERS = {
    10: "TB3_0",
    11: "TB3_1",
    12: "TB3_2",
    13: "TB3_3",
    14: "TB3_4",
}

# Default output directory (relative to this script's location)
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "files" / "markers"


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
    label_height = 80
    total_height = size + label_height + padding * 2
    total_width = size + padding * 2
    
    # Create white background
    output = np.ones((total_height, total_width), dtype=np.uint8) * 255
    
    # Place marker
    y_start = padding
    x_start = padding
    output[y_start:y_start+size, x_start:x_start+size] = marker
    
    # Determine label if not provided
    if label is None:
        if marker_id in REFERENCE_MARKERS:
            label = REFERENCE_MARKERS[marker_id]
        elif marker_id in TURTLEBOT_MARKERS:
            label = TURTLEBOT_MARKERS[marker_id]
        else:
            label = f"ID: {marker_id}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    # Draw main label
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = (total_width - text_size[0]) // 2
    text_y = size + padding + 35
    cv2.putText(output, label, (text_x, text_y), font, font_scale, 0, thickness)
    
    # Draw ID below label
    id_label = f"(ID: {marker_id})"
    id_font_scale = 0.6
    id_thickness = 2
    id_text_size = cv2.getTextSize(id_label, font, id_font_scale, id_thickness)[0]
    id_text_x = (total_width - id_text_size[0]) // 2
    id_text_y = size + padding + 65
    cv2.putText(output, id_label, (id_text_x, id_text_y), font, id_font_scale, 80, id_thickness)
    
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


def generate_a4_marker(
    marker_id: int,
    marker_size_mm: int = 180,
    dictionary_type: int = cv2.aruco.DICT_4X4_50,
    dpi: int = 300
) -> np.ndarray:
    """
    Generate a single marker on an A4 page, filling as much space as possible.
    
    Args:
        marker_id: Marker ID
        marker_size_mm: Physical marker size in millimeters (default: 180mm to fill A4 width)
        dictionary_type: ArUco dictionary type
        dpi: Dots per inch for printing
        
    Returns:
        A4-sized image with centered marker
    """
    # Convert mm to pixels at given DPI
    marker_size_px = int(marker_size_mm * dpi / 25.4)
    
    # Ensure marker fits on A4 with some margin for labels
    max_width = A4_WIDTH_PX - 200  # Leave margin
    max_height = A4_HEIGHT_PX - 600  # Leave space for labels
    marker_size_px = min(marker_size_px, max_width, max_height)
    
    # Generate marker
    marker = generate_marker(marker_id, marker_size_px, dictionary_type)
    
    # Create A4 white background
    a4_page = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype=np.uint8) * 255
    
    # Calculate centering position (vertically centered, with slight offset for labels below)
    x_offset = (A4_WIDTH_PX - marker_size_px) // 2
    y_offset = (A4_HEIGHT_PX - marker_size_px) // 2 - 150  # Slight shift up for label space
    
    # Place marker
    a4_page[y_offset:y_offset+marker_size_px, x_offset:x_offset+marker_size_px] = marker
    
    # Determine label
    if marker_id in REFERENCE_MARKERS:
        label = REFERENCE_MARKERS[marker_id]
        full_name = {
            "TL": "Top Left",
            "TR": "Top Right",
            "BL": "Bottom Left",
            "BR": "Bottom Right"
        }.get(label, label)
    elif marker_id in TURTLEBOT_MARKERS:
        label = TURTLEBOT_MARKERS[marker_id]
        full_name = f"TurtleBot {marker_id - 10}"
    else:
        label = f"ID: {marker_id}"
        full_name = ""
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_y_start = y_offset + marker_size_px + 50
    
    # Draw main label (large)
    main_font_scale = 5.0
    main_thickness = 12
    text_size = cv2.getTextSize(label, font, main_font_scale, main_thickness)[0]
    text_x = (A4_WIDTH_PX - text_size[0]) // 2
    text_y = label_y_start + 100
    cv2.putText(a4_page, label, (text_x, text_y), font, main_font_scale, 0, main_thickness)
    
    # Draw full name below
    if full_name:
        name_font_scale = 2.5
        name_thickness = 5
        name_size = cv2.getTextSize(full_name, font, name_font_scale, name_thickness)[0]
        name_x = (A4_WIDTH_PX - name_size[0]) // 2
        name_y = text_y + 100
        cv2.putText(a4_page, full_name, (name_x, name_y), font, name_font_scale, 40, name_thickness)
    
    # Draw ID at bottom
    id_label = f"ArUco ID: {marker_id}"
    id_font_scale = 1.5
    id_thickness = 3
    id_size = cv2.getTextSize(id_label, font, id_font_scale, id_thickness)[0]
    id_x = (A4_WIDTH_PX - id_size[0]) // 2
    id_y = A4_HEIGHT_PX - 100
    cv2.putText(a4_page, id_label, (id_x, id_y), font, id_font_scale, 80, id_thickness)
    
    # Calculate actual printed size for info
    actual_size_mm = int(marker_size_px * 25.4 / dpi)
    size_label = f"Marker size: {actual_size_mm}mm"
    size_font_scale = 1.0
    size_thickness = 2
    size_size = cv2.getTextSize(size_label, font, size_font_scale, size_thickness)[0]
    size_x = (A4_WIDTH_PX - size_size[0]) // 2
    size_y = A4_HEIGHT_PX - 80
    cv2.putText(a4_page, size_label, (size_x, size_y), font, size_font_scale, 120, size_thickness)
    
    return a4_page


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco markers for TurtleBot localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Generate default reference markers (PDF):
        python generate_markers.py
    
    Generate as PNG instead of PDF:
        python generate_markers.py --format png
    
    Generate specific markers:
        python generate_markers.py --ids 0 1 2 3 10 11 12
    
    Generate TurtleBot markers:
        python generate_markers.py --ids 10 11 12
    
    Generate a printable sheet (multiple per page):
        python generate_markers.py --sheet
    
    Custom marker size (in mm):
        python generate_markers.py --size-mm 150

Default markers (reference corners):
    0 = TL (Top Left)
    1 = TR (Top Right)
    2 = BL (Bottom Left)
    3 = BR (Bottom Right)

TurtleBot markers:
    10 = TB3_0, 11 = TB3_1, 12 = TB3_2, etc.
        """
    )
    
    parser.add_argument("--ids", type=int, nargs="+", default=None,
                       help="Marker IDs to generate (default: 0-3 reference markers)")
    parser.add_argument("--format", type=str, choices=["pdf", "png", "both"], default="pdf",
                       help="Output format: pdf, png, or both (default: pdf)")
    parser.add_argument("--size-mm", type=int, default=180,
                       help="Marker size in millimeters for A4 output (default: 180, max ~190 for A4)")
    parser.add_argument("--size", type=int, default=200,
                       help="Marker size in pixels for sheet mode (default: 200)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sheet", action="store_true",
                       help="Generate a single sheet with all markers (non-A4)")
    parser.add_argument("--per-row", type=int, default=4,
                       help="Markers per row in sheet mode (default: 4)")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for A4 output (default: 300)")
    
    args = parser.parse_args()
    
    # Check PDF support
    if args.format in ["pdf", "both"] and not PIL_AVAILABLE:
        print("Warning: PIL not installed. Install with: pip install Pillow")
        print("Falling back to PNG format.")
        args.format = "png"
    
    # Default marker IDs (reference markers only)
    if args.ids is None:
        args.ids = [0, 1, 2, 3]
        
    print(f"Generating markers: {args.ids}")
    print(f"Output format: {args.format}")
    
    if args.sheet:
        # Generate single sheet (non-A4, compact)
        print(f"Mode: Compact sheet")
        print(f"Size: {args.size}px per marker")
        
        sheet = generate_marker_sheet(
            args.ids,
            markers_per_row=args.per_row,
            marker_size=args.size
        )
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.format in ["png", "both"]:
            output_path = str(output_dir / "markers_sheet.png")
            cv2.imwrite(output_path, sheet)
            print(f"Saved marker sheet: {output_path}")
        
        if args.format in ["pdf", "both"]:
            output_path = str(output_dir / "markers_sheet.pdf")
            img_pil = Image.fromarray(sheet)
            img_pil.save(output_path, "PDF", resolution=args.dpi)
            print(f"Saved marker sheet: {output_path}")
        
    else:
        # Generate individual A4 markers
        print(f"Mode: A4 pages (one marker per page)")
        print(f"Marker size: {args.size_mm}mm")
        print(f"DPI: {args.dpi}")
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all marker images for combined PDF
        marker_images: List[Image.Image] = []
        
        for marker_id in args.ids:
            # Generate A4 page with marker
            a4_marker = generate_a4_marker(
                marker_id,
                marker_size_mm=args.size_mm,
                dpi=args.dpi
            )
            
            # Determine filename based on marker type
            if marker_id in REFERENCE_MARKERS:
                label = REFERENCE_MARKERS[marker_id]
                base_filename = f"ref_{label}_id{marker_id:02d}"
            elif marker_id in TURTLEBOT_MARKERS:
                label = TURTLEBOT_MARKERS[marker_id]
                base_filename = f"tb3_{marker_id - 10}_id{marker_id:02d}"
            else:
                base_filename = f"marker_id{marker_id:02d}"
            
            # Save PNG if requested
            if args.format in ["png", "both"]:
                output_path = output_dir / f"{base_filename}.png"
                cv2.imwrite(str(output_path), a4_marker)
                print(f"  Saved: {output_path}")
            
            # Collect for PDF
            if args.format in ["pdf", "both"]:
                img_pil = Image.fromarray(a4_marker)
                marker_images.append(img_pil)
        
        # Save combined PDF with all markers
        if args.format in ["pdf", "both"] and marker_images:
            pdf_path = output_dir / "markers_all.pdf"
            marker_images[0].save(
                str(pdf_path),
                "PDF",
                resolution=args.dpi,
                save_all=True,
                append_images=marker_images[1:] if len(marker_images) > 1 else []
            )
            print(f"  Saved combined PDF: {pdf_path}")
        
        print(f"\nGenerated {len(args.ids)} A4 marker pages in {output_dir}")
    
    print("\nPrinting tips:")
    print("  - Print at 100% scale (no fit-to-page)")
    print("  - Use matte paper to reduce glare")
    print("  - Ensure markers are flat when mounted")
    print(f"  - Physical marker size: {args.size_mm}mm")


if __name__ == "__main__":
    main()
