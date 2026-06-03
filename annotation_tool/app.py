import os
import sys
import yaml
import json
import threading
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, send_from_directory, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

# Base paths
WORKSPACE_DIR = "/home/ansyah/TA-main"
DEFAULT_YOLO_DIR = os.path.join(WORKSPACE_DIR, "TA_Lite", "yolo_dataset")
DATASET_YAML_PATH = os.path.join(DEFAULT_YOLO_DIR, "dataset.yaml")

# Supported image extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

PROJECTS_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects.json")

def load_projects():
    if os.path.exists(PROJECTS_JSON_PATH):
        try:
            with open(PROJECTS_JSON_PATH, 'r') as f:
                data = json.load(f)
                if "projects" in data and "active_id" in data:
                    return data
        except Exception as e:
            print(f"Error loading projects.json: {e}")
            
    # Create default project pointing to TA_Lite dataset directory
    default_classes = ["car", "motor"]
    if os.path.exists(DATASET_YAML_PATH):
        try:
            with open(DATASET_YAML_PATH, 'r') as f:
                yaml_data = yaml.safe_load(f)
                names = yaml_data.get('names', {0: "car", 1: "motor"})
                default_classes = [names[k] for k in sorted(names.keys())]
        except Exception as e:
            print(f"Error loading default dataset.yaml: {e}")
            
    default_proj = {
        "id": "default",
        "name": "Default Project (TA_Lite)",
        "images_dir": os.path.join(DEFAULT_YOLO_DIR, "images"),
        "labels_dir": os.path.join(DEFAULT_YOLO_DIR, "labels"),
        "classes": default_classes,
        "format": "yolo"
    }
    
    data = {
        "projects": [default_proj],
        "active_id": "default"
    }
    save_projects(data)
    return data

def save_projects(data):
    try:
        with open(PROJECTS_JSON_PATH, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving projects.json: {e}")
        
def get_active_project():
    data = load_projects()
    active_id = data.get("active_id", "default")
    for proj in data.get("projects", []):
        if proj.get("id") == active_id:
            return proj
    if data.get("projects"):
        return data["projects"][0]
    return None

def load_dataset_config():
    proj = get_active_project()
    if proj:
        # Return classes dictionary: {index: name}
        return {i: name for i, name in enumerate(proj.get("classes", []))}
    return {0: "car", 1: "motor"}

def get_yolo_model():
    """Finds the best available YOLO model: custom trained first, then fallback to pretrained."""
    from ultralytics import YOLO
    
    possible_paths = [
        os.path.join(WORKSPACE_DIR, "TA_Lite", "yolo_output", "vehicle_detector", "weights", "best.pt"),
        os.path.join(WORKSPACE_DIR, "TA_Lite", "runs", "detect", "vehicle_detector", "weights", "best.pt"),
        os.path.join(WORKSPACE_DIR, "TA_Lite", "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(WORKSPACE_DIR, "TA_Lite", "yolov8n.pt"),
        "yolov8n.pt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[Model] Loading model from: {path}")
            return YOLO(path), path
            
    print("[Model] Using standard yolov8n.pt model")
    return YOLO("yolov8n.pt"), "yolov8n.pt"

def load_voc_annotations(label_path, classes):
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return []
    try:
        tree = ET.parse(label_path)
        root = tree.getroot()
        size_el = root.find("size")
        width = float(size_el.find("width").text) if size_el is not None and size_el.find("width") is not None else 1
        height = float(size_el.find("height").text) if size_el is not None and size_el.find("height") is not None else 1
        
        boxes = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            try:
                class_id = classes.index(name)
            except ValueError:
                try:
                    class_id = int(name)
                except:
                    class_id = 0
            
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w / 2
            cy = ymin + h / 2
            
            boxes.append({
                "class_id": class_id,
                "x_center": cx / width if width else 0.5,
                "y_center": cy / height if height else 0.5,
                "width": w / width if width else 0.1,
                "height": h / height if height else 0.1
            })
        return boxes
    except Exception as e:
        print(f"Error loading VOC XML: {e}")
        return []

def save_voc_annotations(image_path, label_path, boxes, width, height, classes):
    root = ET.Element("annotation")
    
    folder = ET.SubElement(root, "folder")
    folder.text = os.path.basename(os.path.dirname(image_path))
    
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)
    
    path = ET.SubElement(root, "path")
    path.text = image_path
    
    size = ET.SubElement(root, "size")
    w = ET.SubElement(size, "width")
    w.text = str(width)
    h = ET.SubElement(size, "height")
    h.text = str(height)
    d = ET.SubElement(size, "depth")
    d.text = "3"
    
    for box in boxes:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        cls_id = int(box.get("class_id", 0))
        name.text = classes[cls_id] if cls_id < len(classes) else str(cls_id)
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        cx = float(box["x_center"]) * width
        cy = float(box["y_center"]) * height
        bw = float(box["width"]) * width
        bh = float(box["height"]) * height
        
        xmin = int(max(0, cx - bw / 2))
        ymin = int(max(0, cy - bh / 2))
        xmax = int(min(width, cx + bw / 2))
        ymax = int(min(height, cy + bh / 2))
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin_el = ET.SubElement(bndbox, "xmin")
        xmin_el.text = str(xmin)
        ymin_el = ET.SubElement(bndbox, "ymin")
        ymin_el.text = str(ymin)
        xmax_el = ET.SubElement(bndbox, "xmax")
        xmax_el.text = str(xmax)
        ymax_el = ET.SubElement(bndbox, "ymax")
        ymax_el.text = str(ymax)
        
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="    ")
    except:
        pass
    tree.write(label_path, encoding="utf-8", xml_declaration=False)

def load_json_annotations(label_path):
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return []
    try:
        with open(label_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

def save_json_annotations(label_path, boxes):
    try:
        with open(label_path, 'w') as f:
            json.dump(boxes, f, indent=4)
    except Exception as e:
        print(f"Error saving JSON: {e}")



@app.route('/')
def index():
    return render_template('index.html')

def get_active_label_ext():
    proj = get_active_project()
    fmt = proj.get("format", "yolo") if proj else "yolo"
    if fmt == "voc":
        return ".xml"
    elif fmt == "json":
        return ".json"
    return ".txt"

@app.route('/api/config')
def get_config():
    proj = get_active_project()
    classes = load_dataset_config()
    sets = []
    if proj:
        images_dir = proj.get("images_dir")
        if os.path.exists(images_dir):
            for name in os.listdir(images_dir):
                if os.path.isdir(os.path.join(images_dir, name)):
                    sets.append(name)
    if not sets:
        sets = ["train", "val"]
    
    return jsonify({
        "classes": classes,
        "sets": sorted(sets),
        "project_name": proj.get("name") if proj else "Default Project",
        "format": proj.get("format", "yolo") if proj else "yolo",
        "class_shortcuts": proj.get("class_shortcuts", {}) if proj else {},
        "class_colors": proj.get("class_colors", {}) if proj else {}
    })

def build_dir_tree(root_dir, current_dir=""):
    """
    Recursively builds a tree representing directories.
    """
    full_path = os.path.join(root_dir, current_dir)
    if not os.path.isdir(full_path):
        return None
        
    node = {
        "name": os.path.basename(current_dir) if current_dir else "images",
        "relative_path": current_dir,
        "children": []
    }
    
    try:
        # Sort files/folders to be deterministic
        for item in sorted(os.listdir(full_path)):
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                rel_path = os.path.join(current_dir, item) if current_dir else item
                child_node = build_dir_tree(root_dir, rel_path)
                if child_node:
                    node["children"].append(child_node)
    except Exception as e:
        print(f"Error scanning directory {full_path}: {e}")
        
    return node

@app.route('/api/tree')
def get_directory_tree():
    proj = get_active_project()
    if not proj:
        return jsonify({"error": "No active project"}), 400
    images_root = proj.get("images_dir")
    if not os.path.exists(images_root):
        os.makedirs(images_root, exist_ok=True)
    tree = build_dir_tree(images_root)
    return jsonify(tree)

@app.route('/api/images/<path:subpath>')
def get_images(subpath):
    proj = get_active_project()
    if not proj:
        return jsonify({"error": "No active project"}), 400
    images_dir = os.path.join(proj.get("images_dir"), subpath)
    labels_dir = os.path.join(proj.get("labels_dir"), subpath)
    
    if not os.path.exists(images_dir):
        return jsonify({"error": f"Images directory not found for subpath '{subpath}'"}), 404
        
    image_files = []
    ext = get_active_label_ext()
    fmt = proj.get("format", "yolo")
    classes = proj.get("classes", [])
    
    try:
        for f in sorted(os.listdir(images_dir)):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                basename = os.path.splitext(f)[0]
                label_file = basename + ext
                label_path = os.path.join(labels_dir, label_file)
                
                status = "unannotated"
                box_count = 0
                if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                    status = "annotated"
                    try:
                        if fmt == "voc":
                            boxes = load_voc_annotations(label_path, classes)
                            box_count = len(boxes)
                        elif fmt == "json":
                            boxes = load_json_annotations(label_path)
                            box_count = len(boxes)
                        else:
                            with open(label_path, 'r') as lf:
                                box_count = len([line for line in lf if line.strip()])
                    except Exception as e:
                        print(f"Error reading labels: {e}")
                elif os.path.exists(label_path):
                    status = "semi-annotated" # created but empty
                    
                image_files.append({
                    "filename": f,
                    "status": status,
                    "box_count": box_count
                })
    except Exception as e:
         return jsonify({"error": str(e)}), 500
         
    return jsonify({"images": image_files})

from werkzeug.utils import secure_filename

@app.route('/api/image/<path:subpath>/<filename>')
def serve_image(subpath, filename):
    proj = get_active_project()
    if not proj:
        return jsonify({"error": "No active project"}), 400
    images_dir = os.path.join(proj.get("images_dir"), subpath)
    return send_from_directory(images_dir, filename)

@app.route('/api/upload/<path:subpath>', methods=['POST'])
def upload_image(subpath):
    proj = get_active_project()
    if not proj:
        return jsonify({"error": "No active project"}), 400
        
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    relative_path = request.form.get('relativePath', '')
    
    if file and file.filename.lower().endswith(IMAGE_EXTENSIONS):
        filename = secure_filename(os.path.basename(file.filename))
        
        # Resolve target nested directories if relative_path is provided
        if relative_path:
            path_parts = [secure_filename(p) for p in os.path.dirname(relative_path).split('/') if p]
            sub_dir = os.path.join(*path_parts) if path_parts else ""
            images_dir = os.path.join(proj.get("images_dir"), subpath, sub_dir)
            labels_dir = os.path.join(proj.get("labels_dir"), subpath, sub_dir)
        else:
            images_dir = os.path.join(proj.get("images_dir"), subpath)
            labels_dir = os.path.join(proj.get("labels_dir"), subpath)
            
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        dest_path = os.path.join(images_dir, filename)
        
        # Avoid name collision
        basename, ext_img = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dest_path):
            filename = f"{basename}_{counter}{ext_img}"
            dest_path = os.path.join(images_dir, filename)
            counter += 1
            
        file.save(dest_path)
        
        # Create empty label file in correct format
        label_file = os.path.splitext(filename)[0] + get_active_label_ext()
        label_path = os.path.join(labels_dir, label_file)
        if not os.path.exists(label_path):
            with open(label_path, 'w') as lf:
                pass
                
        final_relative_filename = os.path.join(os.path.dirname(relative_path), filename) if relative_path else filename
        
        return jsonify({
            "success": True,
            "filename": final_relative_filename,
            "status": "unannotated",
            "box_count": 0
        })
        
    return jsonify({"error": "Invalid image format"}), 400

@app.route('/api/delete-folder/<path:subpath>', methods=['DELETE'])
def delete_folder(subpath):
    proj = get_active_project()
    if not proj:
        return jsonify({"error": "No active project"}), 400
        
    images_dir = os.path.join(proj.get("images_dir"), subpath)
    labels_dir = os.path.join(proj.get("labels_dir"), subpath)
    
    # Security check: Ensure we don't delete outside project directory
    images_dir = os.path.abspath(images_dir)
    labels_dir = os.path.abspath(labels_dir)
    proj_images_root = os.path.abspath(proj.get("images_dir"))
    proj_labels_root = os.path.abspath(proj.get("labels_dir"))
    
    if not images_dir.startswith(proj_images_root) or images_dir == proj_images_root:
        return jsonify({"error": "Invalid directory path or cannot delete root directory"}), 400
        
    import shutil
    try:
        deleted_images = False
        deleted_labels = False
        
        if os.path.exists(images_dir) and os.path.isdir(images_dir):
            shutil.rmtree(images_dir)
            deleted_images = True
            
        if os.path.exists(labels_dir) and os.path.isdir(labels_dir):
            shutil.rmtree(labels_dir)
            deleted_labels = True
            
        if not deleted_images and not deleted_labels:
            return jsonify({"error": "Folder not found"}), 404
            
        return jsonify({"success": True, "message": f"Successfully deleted folder '{subpath}'"})
    except Exception as e:
        return jsonify({"error": f"Failed to delete folder: {str(e)}"}), 500

@app.route('/api/annotations/<path:subpath>/<filename>', methods=['GET', 'POST'])
def handle_annotations(subpath, filename):
    proj = get_active_project()
    if not proj:
        return jsonify({"error": "No active project"}), 400
        
    labels_dir = os.path.join(proj.get("labels_dir"), subpath)
    os.makedirs(labels_dir, exist_ok=True)
    
    basename = os.path.splitext(filename)[0]
    label_path = os.path.join(labels_dir, basename + get_active_label_ext())
    
    fmt = proj.get("format", "yolo")
    classes = proj.get("classes", [])
    
    if request.method == 'GET':
        boxes = []
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            try:
                if fmt == "voc":
                    boxes = load_voc_annotations(label_path, classes)
                elif fmt == "json":
                    boxes = load_json_annotations(label_path)
                else:
                    # YOLO
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                boxes.append({
                                    "class_id": int(parts[0]),
                                    "x_center": float(parts[1]),
                                    "y_center": float(parts[2]),
                                    "width": float(parts[3]),
                                    "height": float(parts[4])
                                })
            except Exception as e:
                return jsonify({"error": f"Failed to read labels: {str(e)}"}), 500
        return jsonify({"boxes": boxes})
        
    elif request.method == 'POST':
        data = request.json
        boxes = data.get('boxes', [])
        width = data.get('width', 640)
        height = data.get('height', 480)
        
        try:
            if fmt == "voc":
                image_path = os.path.join(proj.get("images_dir"), subpath, filename)
                save_voc_annotations(image_path, label_path, boxes, width, height, classes)
            elif fmt == "json":
                save_json_annotations(label_path, boxes)
            else:
                # YOLO
                with open(label_path, 'w') as f:
                    for box in boxes:
                        f.write(f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}\n")
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": f"Failed to save labels: {str(e)}"}), 500

@app.route('/api/auto-annotate/<path:subpath>/<filename>', methods=['POST'])
def auto_annotate(subpath, filename):
    proj = get_active_project()
    if not proj:
        return jsonify({"error": "No active project"}), 400
        
    image_path = os.path.join(proj.get("images_dir"), subpath, filename)
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404
        
    try:
        model, path = get_yolo_model()
        results = model(image_path, verbose=False)
        
        local_classes = proj.get("classes", [])
        is_coco = "yolov8n.pt" in path or "coco" in path.lower()
        
        coco_mapping = {
            2: 0,  # car
            3: 1,  # motorcycle
            5: 0,  # bus
            7: 0   # truck
        }
        
        boxes = []
        for result in results:
            for box in result.boxes:
                xywhn = box.xywhn[0].tolist()
                model_cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                if conf < 0.25:
                    continue
                    
                target_cls = None
                if is_coco:
                    if model_cls in coco_mapping:
                        mapped_idx = coco_mapping[model_cls]
                        if mapped_idx < len(local_classes):
                            target_cls = mapped_idx
                else:
                    if model_cls < len(local_classes):
                        target_cls = model_cls
                        
                if target_cls is not None:
                    boxes.append({
                        "class_id": target_cls,
                        "x_center": xywhn[0],
                        "y_center": xywhn[1],
                        "width": xywhn[2],
                        "height": xywhn[3],
                        "confidence": conf
                    })
                    
        return jsonify({"boxes": boxes})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Auto-annotation failed: {str(e)}"}), 500

@app.route('/api/auto-annotate-folder/<path:subpath>', methods=['POST'])
def auto_annotate_folder(subpath):
    proj = get_active_project()
    if not proj:
        return jsonify({"error": "No active project"}), 400
        
    images_dir = os.path.join(proj.get("images_dir"), subpath)
    labels_dir = os.path.join(proj.get("labels_dir"), subpath)
    
    if not os.path.exists(images_dir):
        return jsonify({"error": f"Images directory not found for subpath '{subpath}'"}), 404
        
    try:
        ext = get_active_label_ext()
        fmt = proj.get("format", "yolo")
        local_classes = proj.get("classes", [])
        
        # Find all images that are not yet annotated (no label file or label file is empty)
        image_files = []
        for f in sorted(os.listdir(images_dir)):
            if f.lower().endswith(IMAGE_EXTENSIONS):
                basename = os.path.splitext(f)[0]
                label_path = os.path.join(labels_dir, basename + ext)
                if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                    image_files.append(f)
                    
        if not image_files:
            return jsonify({
                "success": True,
                "message": "All images in this folder are already annotated or reviewed.",
                "count": 0
            })
            
        model, path = get_yolo_model()
        is_coco = "yolov8n.pt" in path or "coco" in path.lower()
        
        coco_mapping = {
            2: 0,  # car
            3: 1,  # motorcycle
            5: 0,  # bus
            7: 0   # truck
        }
        
        os.makedirs(labels_dir, exist_ok=True)
        annotated_count = 0
        
        for img_name in image_files:
            image_path = os.path.join(images_dir, img_name)
            results = model(image_path, verbose=False)
            
            # Extract width/height for VOC XML
            width, height = 640, 480
            if fmt == "voc":
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"PIL error in folder auto annotate: {e}")
            
            boxes = []
            for result in results:
                for box in result.boxes:
                    xywhn = box.xywhn[0].tolist()
                    model_cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    if conf < 0.25:
                        continue
                        
                    target_cls = None
                    if is_coco:
                        if model_cls in coco_mapping:
                            mapped_idx = coco_mapping[model_cls]
                            if mapped_idx < len(local_classes):
                                target_cls = mapped_idx
                    else:
                        if model_cls < len(local_classes):
                            target_cls = model_cls
                            
                    if target_cls is not None:
                        boxes.append({
                            "class_id": target_cls,
                            "x_center": xywhn[0],
                            "y_center": xywhn[1],
                            "width": xywhn[2],
                            "height": xywhn[3]
                        })
                        
            # Write to label file
            basename = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, basename + ext)
            if fmt == "voc":
                save_voc_annotations(image_path, label_path, boxes, width, height, local_classes)
            elif fmt == "json":
                save_json_annotations(label_path, boxes)
            else:
                with open(label_path, 'w') as lf:
                    for box in boxes:
                        lf.write(f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}\n")
            annotated_count += 1
            
        return jsonify({
            "success": True,
            "message": f"Successfully auto-annotated {annotated_count} new images.",
            "count": annotated_count
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Folder auto-annotation failed: {str(e)}"}), 500

@app.route('/api/projects', methods=['GET'])
def list_projects():
    data = load_projects()
    return jsonify({
        "projects": data.get("projects", []),
        "active_id": data.get("active_id", "default")
    })

@app.route('/api/projects/active', methods=['POST'])
def set_active_project_route():
    data = load_projects()
    project_id = request.json.get("project_id")
    
    found = False
    for proj in data.get("projects", []):
        if proj.get("id") == project_id:
            found = True
            break
            
    if not found:
        return jsonify({"error": "Project not found"}), 404
        
    data["active_id"] = project_id
    save_projects(data)
    return jsonify({"success": True})

@app.route('/api/projects/create', methods=['POST'])
def create_project_route():
    data = load_projects()
    req_data = request.json
    
    name = req_data.get("name")
    images_dir = req_data.get("images_dir")
    labels_dir = req_data.get("labels_dir")
    classes = req_data.get("classes", [])
    fmt = req_data.get("format", "yolo")
    class_shortcuts_input = req_data.get("class_shortcuts", [])
    
    if not name or not images_dir or not labels_dir:
        return jsonify({"error": "Project name, images directory, and labels directory are required."}), 400
        
    import uuid
    project_id = str(uuid.uuid4())[:8]
    
    class_shortcuts = {}
    for idx, key in enumerate(class_shortcuts_input):
        if key.strip():
            class_shortcuts[str(idx)] = key.strip()
            
    new_proj = {
        "id": project_id,
        "name": name,
        "images_dir": os.path.abspath(images_dir),
        "labels_dir": os.path.abspath(labels_dir),
        "classes": [c.strip() for c in classes if c.strip()],
        "format": fmt,
        "class_shortcuts": class_shortcuts
    }
    
    data["projects"].append(new_proj)
    data["active_id"] = project_id
    save_projects(data)
    
    os.makedirs(new_proj["images_dir"], exist_ok=True)
    os.makedirs(new_proj["labels_dir"], exist_ok=True)
    
    return jsonify({"success": True, "project": new_proj})

@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project_route(project_id):
    if project_id == "default":
        return jsonify({"error": "Cannot delete the default project."}), 400
        
    data = load_projects()
    projects = data.get("projects", [])
    new_projects = [p for p in projects if p.get("id") != project_id]
    
    if len(projects) == len(new_projects):
        return jsonify({"error": "Project not found"}), 404
        
    data["projects"] = new_projects
    if data.get("active_id") == project_id:
        data["active_id"] = "default"
        
    save_projects(data)
    return jsonify({"success": True})

@app.route('/api/projects/update', methods=['POST'])
def update_project_route():
    data = load_projects()
    req_data = request.json
    project_id = req_data.get("project_id")
    
    proj = None
    for p in data.get("projects", []):
        if p.get("id") == project_id:
            proj = p
            break
            
    if not proj:
        return jsonify({"error": "Project not found"}), 404
        
    shortcuts = req_data.get("class_shortcuts", {})
    proj["class_shortcuts"] = shortcuts
    
    colors = req_data.get("class_colors", {})
    proj["class_colors"] = colors
    
    save_projects(data)
    return jsonify({"success": True, "project": proj})

# Global variables for training session
training_thread = None
training_process = None
training_logs = []
training_status = "idle"  # "idle", "training", "completed", "error"
training_device = "Unknown"

def training_worker(project, epochs, batch, imgsz):
    global training_status, training_logs, training_process, training_device
    
    import subprocess
    import shutil
    
    try:
        training_status = "training"
        training_logs = ["--- Starting YOLOv8 Training Session ---"]
        
        # Check PyTorch device availability
        try:
            import torch
            if torch.cuda.is_available():
                device_type = "cuda"
                training_device = f"GPU: {torch.cuda.get_device_name(0)}"
            else:
                device_type = "cpu"
                training_device = "CPU (Slow)"
        except ImportError:
            device_type = "cpu"
            training_device = "CPU (PyTorch not found)"
            
        training_logs.append(f"Hardware Detection: Using {training_device}")
        
        # 1. Prepare YAML config and file lists recursively
        images_dir = project["images_dir"]
        labels_dir = project["labels_dir"]
        parent_dir = os.path.dirname(images_dir)
        yaml_path = os.path.join(parent_dir, "dataset_training.yaml")
        
        # Recursively find all images that have annotated labels
        all_image_paths = []
        total_images_found = 0
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    total_images_found += 1
                    full_img_path = os.path.join(root, file)
                    # Verify if the corresponding label file exists and contains boxes
                    rel_to_images = os.path.relpath(full_img_path, images_dir)
                    label_file = os.path.splitext(rel_to_images)[0] + ".txt"
                    full_lbl_path = os.path.join(labels_dir, label_file)
                    
                    if os.path.exists(full_lbl_path) and os.path.getsize(full_lbl_path) > 0:
                        # Use absolute path to ensure correct label mapping by ultralytics YOLOv8
                        all_image_paths.append(os.path.abspath(full_img_path))

        if not all_image_paths:
            training_status = "error"
            training_logs.append("Error: No annotated images found! Please draw bounding boxes on at least one image before starting training.")
            return

        training_logs.append(f"Found {total_images_found} total images, with {len(all_image_paths)} annotated images across all folders (including subfolders).")

        # Create train and validation image list files
        train_txt_path = os.path.join(parent_dir, "train_images.txt")
        val_txt_path = os.path.join(parent_dir, "val_images.txt")
        
        import random
        random.seed(42)
        shuffled_paths = list(all_image_paths)
        random.shuffle(shuffled_paths)
        
        if len(shuffled_paths) >= 10:
            split_idx = int(len(shuffled_paths) * 0.9)
            train_set = shuffled_paths[:split_idx]
            val_set = shuffled_paths[split_idx:]
        else:
            train_set = shuffled_paths
            val_set = shuffled_paths
            
        with open(train_txt_path, 'w') as f:
            for p in train_set:
                f.write(p + '\n')
                
        with open(val_txt_path, 'w') as f:
            for p in val_set:
                f.write(p + '\n')

        training_logs.append(f"Split dataset: {len(train_set)} train images, {len(val_set)} validation images.")
        
        # Clean existing YOLO cache files in labels directory to avoid path mismatch issues
        cleaned_cache = False
        for root_dir, _, files in os.walk(labels_dir):
            for file in files:
                if file.endswith(".cache"):
                    cache_file_path = os.path.join(root_dir, file)
                    try:
                        os.remove(cache_file_path)
                        cleaned_cache = True
                    except Exception as ce:
                        training_logs.append(f"Warning: Could not clear cache file {file}: {ce}")
        if cleaned_cache:
            training_logs.append("Cleared existing YOLO dataset cache files to force refresh.")
        
        yaml_data = {
            "path": parent_dir,
            "train": "train_images.txt",
            "val": "val_images.txt",
            "names": {i: name for i, name in enumerate(project["classes"])}
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
            
        training_logs.append(f"Saved dataset configuration to: {yaml_path}")
        
        # We will use yolov8n.pt as our base pretrained model
        # Make sure runs/detect/train exists and is writable
        runs_dir = os.path.join(WORKSPACE_DIR, "TA_Lite", "runs", "detect")
        os.makedirs(runs_dir, exist_ok=True)
        
        # Format the inline python code to train
        # Force detected device type and limit worker count to 2 to optimize resource usage
        python_code = (
            f"from ultralytics import YOLO; "
            f"model = YOLO('yolov8n.pt'); "
            f"model.train(data='{yaml_path}', epochs={epochs}, batch={batch}, imgsz={imgsz}, "
            f"project='{runs_dir}', name='train', exist_ok=True, "
            f"device='{device_type}', workers=2)"
        )
        
        cmd = [
            sys.executable,
            "-u",
            "-c",
            python_code
        ]
        
        training_logs.append(f"Spawning PyTorch YOLO training subprocess...")
        
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=WORKSPACE_DIR
        )
        
        for line in iter(training_process.stdout.readline, ''):
            if not line:
                break
            training_logs.append(line.rstrip())
            if len(training_logs) > 1000:
                training_logs.pop(1)
                
        training_process.stdout.close()
        return_code = training_process.wait()
        
        if return_code == 0:
            training_status = "completed"
            training_logs.append("--- Training Completed Successfully! ---")
            
            # Export to ONNX
            best_pt = os.path.join(runs_dir, "train", "weights", "best.pt")
            if os.path.exists(best_pt):
                training_logs.append("Found best.pt weights. Exporting to ONNX...")
                try:
                    from ultralytics import YOLO
                    model = YOLO(best_pt)
                    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=12)
                    
                    # Copy to output/yolov8n_vehicle.onnx
                    dest_onnx = os.path.join(WORKSPACE_DIR, "TA_Lite", "output", "yolov8n_vehicle.onnx")
                    os.makedirs(os.path.dirname(dest_onnx), exist_ok=True)
                    shutil.copy(onnx_path, dest_onnx)
                    training_logs.append(f"Exported model successfully saved to: {dest_onnx}")
                except Exception as ex:
                    training_logs.append(f"Error exporting to ONNX: {ex}")
            else:
                training_logs.append("Warning: best.pt weights not found in output directory!")
        else:
            training_status = "error"
            training_logs.append(f"--- Training Process Terminated / Failed (Exit Code: {return_code}) ---")
            
    except Exception as e:
        training_status = "error"
        training_logs.append(f"Exception in training worker thread: {e}")
    finally:
        training_process = None

@app.route('/api/train/start', methods=['POST'])
def start_training():
    global training_thread, training_status, training_logs
    
    if training_status == "training":
        return jsonify({"error": "Training is already in progress."}), 400
        
    req_data = request.json or {}
    epochs = int(req_data.get("epochs", 10))
    batch = int(req_data.get("batch", 8))
    imgsz = int(req_data.get("imgsz", 320))
    
    # Load active project
    projects_data = load_projects()
    active_id = projects_data.get("active_id")
    active_proj = None
    for p in projects_data.get("projects", []):
        if p.get("id") == active_id:
            active_proj = p
            break
            
    if not active_proj:
        return jsonify({"error": "No active project loaded."}), 400
        
    # Start thread
    training_thread = threading.Thread(
        target=training_worker,
        args=(active_proj, epochs, batch, imgsz),
        daemon=True
    )
    training_thread.start()
    
    return jsonify({"success": True, "message": "Training started."})

@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    global training_status, training_logs, training_device
    return jsonify({
        "status": training_status,
        "logs": training_logs,
        "device": training_device
    })

@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    global training_process, training_status, training_logs
    
    if training_process:
        try:
            training_process.terminate()
            training_logs.append("--- Training manually stopped by user ---")
            training_status = "idle"
            return jsonify({"success": True, "message": "Training process terminated."})
        except Exception as e:
            return jsonify({"error": f"Failed to stop training: {e}"}), 500
            
    return jsonify({"success": True, "message": "No active training process running."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

