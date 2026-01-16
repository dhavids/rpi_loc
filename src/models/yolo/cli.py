#!/usr/bin/env python3
"""
CLI for YOLO TurtleBot Detection Training

Provides command-line interface for:
- Collecting training images
- Annotating images
- Creating datasets
- Training/fine-tuning models
- Evaluating models

Usage:
    yolo_loc collect --stream --output data/images
    yolo_loc annotate --images data/images --labels data/labels
    yolo_loc create-dataset --images data/images --labels data/labels --output data/dataset
    yolo_loc train --data data/dataset --epochs 100
    yolo_loc finetune --data data/dataset --base-model yolov8n.pt --epochs 50
    yolo_loc evaluate --model runs/train/best.pt --data data/dataset
    yolo_loc predict --model runs/train/best.pt --source test_images/
    yolo_loc export --model runs/train/best.pt --format onnx
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths relative to this file's location
# Structure: rpi_loc/src/models/yolo/cli.py -> rpi_loc/files/models/yolo/
_THIS_DIR = Path(__file__).parent
_RPI_LOC_DIR = _THIS_DIR.parent.parent.parent
_FILES_DIR = _RPI_LOC_DIR / "files" / "models" / "yolo"
DEFAULT_DATA_DIR = _FILES_DIR / "data"
DEFAULT_RUNS_DIR = _FILES_DIR / "runs"
DEFAULT_IMAGES_DIR = DEFAULT_DATA_DIR / "images"
DEFAULT_LABELS_DIR = DEFAULT_DATA_DIR / "labels"
DEFAULT_DATASET_DIR = DEFAULT_DATA_DIR / "dataset"


def _resolve_path(user_path: str) -> str:
    """
    Resolve a user-provided path, checking files directory if not found in pwd.
    
    For a path like 'data/images_4':
    1. If it exists relative to pwd, use it
    2. If parent (data) exists relative to pwd, use pwd path
    3. If parent (data) exists in files directory, use files/data/images_4
    4. Otherwise, default to pwd path
    
    Returns the resolved absolute path as a string.
    """
    path = Path(user_path)
    
    # If absolute path, use as-is
    if path.is_absolute():
        return str(path)
    
    cwd = Path.cwd()
    cwd_path = cwd / path
    
    # If the path or its parent exists in pwd, use pwd
    if cwd_path.exists() or cwd_path.parent.exists():
        return str(cwd_path)
    
    # Check if the first component (e.g., 'data') exists in files directory
    parts = path.parts
    if parts:
        first_part = parts[0]
        files_base = _RPI_LOC_DIR / "files"
        
        # Check if first_part exists in files directory
        if (files_base / first_part).exists():
            # Use files directory path
            resolved = files_base / path
            logger.info(f"Resolved '{user_path}' to files directory: {resolved}")
            return str(resolved)
    
    # Default: use cwd path
    return str(cwd_path)


def _get_default_path(path_type: str) -> str:
    """Get default path, creating directory if needed."""
    paths = {
        "images": DEFAULT_IMAGES_DIR,
        "labels": DEFAULT_LABELS_DIR,
        "dataset": DEFAULT_DATASET_DIR,
        "data": DEFAULT_DATA_DIR,
        "runs": DEFAULT_RUNS_DIR,
    }
    path = paths.get(path_type, DEFAULT_DATA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def cmd_collect(args):
    """Collect images for training."""
    from .collector import ImageCollector, StreamCollector, CollectorConfig
    
    config = CollectorConfig(
        output_dir=args.output,
        auto_capture_interval=args.interval,
        max_images=args.max_images
    )
    
    if args.stream:
        # Collect from RPi camera stream
        collector = StreamCollector(
            host=args.host,
            port=args.port,
            config=config
        )
        if args.duration:
            count = collector.collect_timed(
                duration=args.duration,
                interval=args.interval or 2.0,
                show_preview=args.preview
            )
        else:
            count = collector.collect_interactive()
    elif args.video:
        collector = ImageCollector(source=args.video, config=config)
        count = collector.extract_frames(every_n=args.every_n)
    else:
        collector = ImageCollector(source=args.camera, config=config)
        count = collector.collect_interactive()
    
    print(f"\nCollected {count} images to {args.output}")


def cmd_annotate(args):
    """Annotate images with bounding boxes."""
    from .annotator import ImageAnnotator
    
    class_names = args.classes.split(",") if args.classes else ["turtlebot"]
    
    annotator = ImageAnnotator(
        images_dir=args.images,
        labels_dir=args.labels,
        class_names=class_names
    )
    
    if args.progress:
        progress = annotator.get_progress()
        print(f"\nAnnotation Progress:")
        print(f"  Total images: {progress['total_images']}")
        print(f"  Annotated: {progress['annotated']}")
        print(f"  Remaining: {progress['remaining']}")
        print(f"  Total boxes: {progress['total_boxes']}")
        print(f"  Progress: {progress['progress_percent']:.1f}%")
    else:
        annotator.run()


def cmd_create_dataset(args):
    """Create dataset from annotated images."""
    from .dataset import DatasetManager
    
    class_names = args.classes.split(",") if args.classes else ["turtlebot"]
    
    manager = DatasetManager(args.output, class_names)
    
    # If no images/labels provided, just create empty structure
    if not args.images or not args.labels:
        manager.create_structure()
        manager.create_data_yaml()
        print(f"\nEmpty dataset structure created at {args.output}")
        print("  Add images to train/images/ and labels to train/labels/")
        return
    
    counts = manager.split_dataset(
        images_folder=args.images,
        labels_folder=args.labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    print(f"\nDataset created at {args.output}")
    print(f"  Train: {counts['train']} images")
    print(f"  Val: {counts['val']} images")
    print(f"  Test: {counts['test']} images")


def cmd_train(args):
    """Train YOLO model."""
    from .trainer import YOLOTrainer, TrainingConfig
    
    config = TrainingConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        name=args.name,
        workers=args.workers
    )
    
    trainer = YOLOTrainer()
    
    # Find data.yaml
    data_path = Path(args.data)
    if data_path.is_dir():
        data_yaml = data_path / "data.yaml"
    else:
        data_yaml = data_path
    
    if not data_yaml.exists():
        print(f"Error: data.yaml not found at {data_yaml}")
        sys.exit(1)
    
    print(f"\nStarting training:")
    print(f"  Base model: {args.base_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}")
    print(f"  Device: {args.device}")
    print()
    
    results = trainer.train(str(data_yaml), config, resume=args.resume)
    
    print(f"\nTraining complete!")
    print(f"  Best model: {results.get('model_path', 'N/A')}")


def cmd_finetune(args):
    """Fine-tune pretrained model."""
    from .trainer import YOLOTrainer
    
    trainer = YOLOTrainer()
    
    data_path = Path(args.data)
    if data_path.is_dir():
        data_yaml = data_path / "data.yaml"
    else:
        data_yaml = data_path
    
    print(f"\nFine-tuning {args.base_model} on {data_yaml}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Freeze layers: {args.freeze_layers}")
    print()
    
    results = trainer.finetune(
        str(data_yaml),
        base_model=args.base_model,
        epochs=args.epochs,
        freeze_layers=args.freeze_layers,
        batch_size=args.batch_size,
        workers=args.workers
    )
    
    print(f"\nFine-tuning complete!")
    print(f"  Best model: {results.get('model_path', 'N/A')}")


def cmd_evaluate(args):
    """Evaluate model on test data."""
    from .trainer import YOLOTrainer
    
    trainer = YOLOTrainer(model_path=args.model)
    
    data_yaml = None
    if args.data:
        data_path = Path(args.data)
        if data_path.is_dir():
            data_yaml = str(data_path / "data.yaml")
        else:
            data_yaml = str(data_path)
    
    print(f"\nEvaluating model: {args.model}")
    
    metrics = trainer.validate(data_yaml)
    
    print(f"\nResults:")
    print(f"  mAP@50: {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")


def cmd_export(args):
    """Export model to different format."""
    from .trainer import YOLOTrainer
    
    trainer = YOLOTrainer(model_path=args.model)
    
    print(f"\nExporting {args.model} to {args.format} format...")
    
    output = trainer.export(format=args.format, output_path=args.output)
    
    print(f"Exported to: {output}")


def cmd_predict(args):
    """Run inference on images."""
    from .trainer import YOLOTrainer
    
    trainer = YOLOTrainer(model_path=args.model)
    
    print(f"\nRunning inference on: {args.source}")
    
    detections = trainer.predict(
        args.source,
        confidence=args.confidence,
        save=not args.no_save
    )
    
    print(f"\nFound {len(detections)} detections")
    for det in detections[:10]:  # Show first 10
        print(f"  {det['class_name']}: {det['confidence']:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO TurtleBot Detection Training CLI",
        prog="yolo_loc"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Collect command
    p_collect = subparsers.add_parser("collect", help="Collect training images")
    p_collect.add_argument("--camera", type=int, default=0, help="Camera ID")
    p_collect.add_argument("--video", type=str, help="Video file (instead of camera)")
    p_collect.add_argument("--stream", action="store_true", help="Collect from RPi stream")
    p_collect.add_argument("--host", type=str, default="100.99.98.1", help="RPi stream host IP")
    p_collect.add_argument("--port", type=int, default=5000, help="RPi stream port")
    p_collect.add_argument("--duration", type=float, help="Auto-collect duration in seconds")
    p_collect.add_argument("--preview", action="store_true", help="Show preview during timed collect")
    p_collect.add_argument("--output", "-o", default=None, help=f"Output directory (default: {DEFAULT_IMAGES_DIR})")
    p_collect.add_argument("--interval", type=float, default=0, help="Auto-capture interval (seconds)")
    p_collect.add_argument("--max-images", type=int, default=0, help="Maximum images to collect")
    p_collect.add_argument("--every-n", type=int, default=30, help="Extract every N frames (video)")
    p_collect.set_defaults(func=cmd_collect)
    
    # Annotate command
    p_annotate = subparsers.add_parser("annotate", help="Annotate images")
    p_annotate.add_argument("--images", "-i", default=None, help=f"Images directory (default: {DEFAULT_IMAGES_DIR})")
    p_annotate.add_argument("--labels", "-l", default=None, help=f"Labels directory (default: {DEFAULT_LABELS_DIR})")
    p_annotate.add_argument("--classes", "-c", default="turtlebot", help="Comma-separated class names")
    p_annotate.add_argument("--progress", action="store_true", help="Show progress only")
    p_annotate.set_defaults(func=cmd_annotate)
    
    # Create dataset command
    p_dataset = subparsers.add_parser("create-dataset", help="Create train/val/test split")
    p_dataset.add_argument("--images", "-i", default=None, help=f"Images directory (default: {DEFAULT_IMAGES_DIR})")
    p_dataset.add_argument("--labels", "-l", default=None, help=f"Labels directory (default: {DEFAULT_LABELS_DIR})")
    p_dataset.add_argument("--output", "-o", default=None, help=f"Output dataset directory (default: {DEFAULT_DATASET_DIR})")
    p_dataset.add_argument("--classes", "-c", default="turtlebot", help="Comma-separated class names")
    p_dataset.add_argument("--train-ratio", type=float, default=0.8)
    p_dataset.add_argument("--val-ratio", type=float, default=0.15)
    p_dataset.add_argument("--test-ratio", type=float, default=0.05)
    p_dataset.add_argument("--no-shuffle", action="store_true")
    p_dataset.add_argument("--seed", type=int, default=42)
    p_dataset.set_defaults(func=cmd_create_dataset)
    
    # Train command
    p_train = subparsers.add_parser("train", help="Train YOLO model")
    p_train.add_argument("--data", "-d", default=None, help=f"Dataset directory or data.yaml (default: {DEFAULT_DATASET_DIR})")
    p_train.add_argument("--base-model", default="yolov8n.pt", help="Base model")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--image-size", type=int, default=640)
    p_train.add_argument("--device", default="auto", help="Device (auto, cpu, cuda, mps)")
    p_train.add_argument("--name", default="turtlebot", help="Run name")
    p_train.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p_train.add_argument("--workers", type=int, default=8, help="Data loader workers (use 0 for Docker)")
    p_train.set_defaults(func=cmd_train)
    
    # Fine-tune command
    p_finetune = subparsers.add_parser("finetune", help="Fine-tune pretrained model")
    p_finetune.add_argument("--data", "-d", default=None, help=f"Dataset directory or data.yaml (default: {DEFAULT_DATASET_DIR})")
    p_finetune.add_argument("--base-model", default="yolov8n.pt", help="Pretrained model")
    p_finetune.add_argument("--epochs", type=int, default=50)
    p_finetune.add_argument("--batch-size", type=int, default=16)
    p_finetune.add_argument("--freeze-layers", type=int, default=0, help="Layers to freeze")
    p_finetune.add_argument("--workers", type=int, default=0, help="Data loader workers (0 for Docker)")
    p_finetune.set_defaults(func=cmd_finetune)
    
    # Evaluate command
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model")
    p_eval.add_argument("--model", "-m", required=True, help="Model path")
    p_eval.add_argument("--data", "-d", default=None, help=f"Dataset directory or data.yaml (default: {DEFAULT_DATASET_DIR})")
    p_eval.set_defaults(func=cmd_evaluate)
    
    # Export command
    p_export = subparsers.add_parser("export", help="Export model")
    p_export.add_argument("--model", "-m", required=True, help="Model path")
    p_export.add_argument("--format", "-f", default="onnx", help="Export format")
    p_export.add_argument("--output", "-o", help="Output path")
    p_export.set_defaults(func=cmd_export)
    
    # Predict command
    p_predict = subparsers.add_parser("predict", help="Run inference")
    p_predict.add_argument("--model", "-m", required=True, help="Model path")
    p_predict.add_argument("--source", "-s", required=True, help="Image/video/directory")
    p_predict.add_argument("--confidence", type=float, default=0.5)
    p_predict.add_argument("--no-save", action="store_true", help="Don't save results")
    p_predict.set_defaults(func=cmd_predict)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Resolve user-provided paths (check files directory if not found in pwd)
    if hasattr(args, 'output') and args.output is not None:
        args.output = _resolve_path(args.output)
    elif hasattr(args, 'output') and args.output is None:
        if args.command == "collect":
            args.output = _get_default_path("images")
        elif args.command == "create-dataset":
            args.output = _get_default_path("dataset")
    
    if hasattr(args, 'images') and args.images is not None:
        args.images = _resolve_path(args.images)
    elif hasattr(args, 'images') and args.images is None:
        args.images = _get_default_path("images")
    
    if hasattr(args, 'labels') and args.labels is not None:
        args.labels = _resolve_path(args.labels)
    elif hasattr(args, 'labels') and args.labels is None:
        args.labels = _get_default_path("labels")
    
    if hasattr(args, 'data') and args.data is not None:
        args.data = _resolve_path(args.data)
    elif hasattr(args, 'data') and args.data is None:
        args.data = _get_default_path("dataset")
    
    if hasattr(args, 'source') and args.source is not None:
        args.source = _resolve_path(args.source)
    
    if hasattr(args, 'model') and args.model is not None:
        args.model = _resolve_path(args.model)
    
    args.func(args)


if __name__ == "__main__":
    main()
