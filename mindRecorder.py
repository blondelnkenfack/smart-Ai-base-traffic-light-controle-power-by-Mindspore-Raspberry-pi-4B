from mindspore.mindrecord import FileWriter
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Class mapping: map name to ID
class_map = {"car": 0, "truck": 1, "bus": 2, "emergency": 3}
# ID mapping: map TXT class index to our internal ID
# In some TXT files, car=0, truck=1, bus=2, emergency=3 or 4.
# We'll assume the same order for now, but handle potential offsets.
id_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3} # mapping TXT class 4 to emergency(3)

def parse_xml(xml_file):
    """Parse Pascal VOC XML annotation file."""
    if not os.path.exists(xml_file):
        return None
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower().strip()
            if name not in class_map:
                continue
            cls = class_map[name]
            bnd = obj.find('bndbox')
            xmin = int(bnd.find('xmin').text)
            ymin = int(bnd.find('ymin').text)
            xmax = int(bnd.find('xmax').text)
            ymax = int(bnd.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax, cls])
        return boxes
    except Exception as e:
        print(f"Error parsing XML {xml_file}: {e}")
        return None

def parse_txt(txt_file, img_w, img_h):
    """Parse YOLO TXT annotation file (normalized coordinates)."""
    if not os.path.exists(txt_file):
        return None
    try:
        boxes = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                # Map TXT ID to our ID
                cls = id_map.get(cls_id, 0)
                
                x_c, y_c, w, h = map(float, parts[1:])
                # Convert normalized back to absolute
                xmin = int((x_c - w/2) * img_w)
                ymin = int((y_c - h/2) * img_h)
                xmax = int((x_c + w/2) * img_w)
                ymax = int((y_c + h/2) * img_h)
                boxes.append([xmin, ymin, xmax, ymax, cls])
        return boxes
    except Exception as e:
        print(f"Error parsing TXT {txt_file}: {e}")
        return None

def create_mindrecord(split_name, base_dir):
    """Convert a dataset split (images + labels) into a MindRecord file."""
    images_dir = Path(base_dir) / split_name / "images"
    labels_dir = Path(base_dir) / split_name / "labels"
    output_file = f"traffic_{split_name}.mindrecord"
    
    if not images_dir.exists():
        print(f"⚠️ Directory not found: {images_dir}. Skipping {split_name} split.")
        return

    print(f"📦 Processing '{split_name}' split...")
    
    # Remove existing mindrecord files to avoid errors
    if os.path.exists(output_file):
        os.remove(output_file)
        if os.path.exists(output_file + ".db"):
            os.remove(output_file + ".db")

    writer = FileWriter(output_file, shard_num=1)
    schema = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]}  # [xmin, ymin, xmax, ymax, class_id]
    }
    writer.add_schema(schema, f"traffic dataset split: {split_name}")

    count = 0
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        # Try XML first
        ann = None
        xml_path = labels_dir / f"{img_path.stem}.xml"
        if xml_path.exists():
            ann = parse_xml(str(xml_path))
        
        # Try TXT if XML failed or doesn't exist
        if ann is None or len(ann) == 0:
            txt_path = labels_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                try:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                    ann = parse_txt(str(txt_path), img_w, img_h)
                except Exception as e:
                    print(f"Error opening image {img_path}: {e}")

        if ann is None or len(ann) == 0:
            # print(f"  ⏭️ Skipping {img_path.name} (no valid labels)")
            continue

        # Convert annotation to numpy array for MindRecord
        ann_np = np.array(ann, dtype=np.int32)

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        writer.write_raw_data([{"image": img_bytes, "annotation": ann_np}])
        count += 1

    if count > 0:
        writer.commit()
        print(f"✅ Created {output_file} with {count} samples.")
    else:
        print(f"❌ No valid samples found for '{split_name}' split.")

if __name__ == "__main__":
    DATASET_BASE = "dataset"
    # Ensure dataset splits exist. If not, try the root directory as a fallback.
    splits = ["train", "valid", "test"]
    for split in splits:
        create_mindrecord(split, DATASET_BASE)
