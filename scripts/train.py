"""Re-export of the training entry-points for convenience."""
from src.train_detector import main as train_detector_main
from src.train_classifier import main as train_classifier_main

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "detector":
        sys.argv.pop(1)
        raise SystemExit(train_detector_main())
    raise SystemExit(train_classifier_main())
