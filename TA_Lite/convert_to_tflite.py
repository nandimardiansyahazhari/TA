import argparse
import sys
import os

def convert_pt_to_tflite(pt_path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' package is not installed. Please install it with 'pip install ultralytics'.")
        return

    print(f"Loading PyTorch model from: {pt_path}")
    try:
        model = YOLO(pt_path)
        print("Exporting model to TFLite format (this will automatically handle YOLOv8 structure)...")
        # Export to tflite (with default resolution 320x320 for mobile performance)
        output_path = model.export(format="tflite", imgsz=320)
        print(f"\n[Success] TFLite model exported to: {output_path}")
    except Exception as e:
        print(f"Export failed: {e}")

def convert_onnx_to_tflite(onnx_path):
    print(f"Converting ONNX model from: {onnx_path} to TFLite...")
    # Generic ONNX to TFLite is best handled by the onnx2tf package to resolve layout transpositions
    try:
        import subprocess
        # Check if onnx2tf is installed
        import onnx2tf
        import tensorflow as tf
        
        output_dir = os.path.splitext(onnx_path)[0] + "_tflite"
        print(f"Running onnx2tf conversion tool. Output will be saved in: {output_dir}")
        cmd = [sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", output_dir]
        subprocess.run(cmd, check=True)
        print(f"\n[Success] Conversion complete! Outputs are generated in: {output_dir}")
    except ImportError:
        print("\nError: Libraries 'onnx2tf' and 'tensorflow' are required for generic ONNX to TFLite conversion.")
        print("Please install them using: pip install tensorflow onnx2tf onnx")
        print("\nRecommendation:")
        print("If this is a YOLOv8 model, it is MUCH easier to convert directly from the PyTorch '.pt' weights file.")
        print("Run: python convert_to_tflite.py --pt path/to/best.pt")
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX or PyTorch weights to TensorFlow Lite (TFLite)")
    parser.add_argument("--onnx", type=str, help="Path to input ONNX file")
    parser.add_argument("--pt", type=str, help="Path to input PyTorch (.pt) file (Recommended for YOLOv8)")
    
    args = parser.parse_args()
    
    if args.pt:
        convert_pt_to_tflite(args.pt)
    elif args.onnx:
        convert_onnx_to_tflite(args.onnx)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python convert_to_tflite.py --pt best.pt")
        print("  python convert_to_tflite.py --onnx best.onnx")
