
import cv2
import numpy as np
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net, context
from model.yolo import YOLOv5

# Configuration
IMAGE_PATH = r"C:/Users/SSD/.gemini/antigravity/brain/44681b2e-673a-4ed3-b15f-84f18a915690/uploaded_image_1768304002800.png"
MODEL_PATH = "yolov5_traffic.ckpt"
INPUT_SIZE = (640, 640)
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = ['Car', 'Truck', 'Emergency'] # Assuming 0=Car

def preprocess(image_path):
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Resize
    h0, w0 = img0.shape[:2]
    r = min(INPUT_SIZE[0] / h0, INPUT_SIZE[1] / w0)
    if r != 1: 
        # Resize preserving aspect ratio (simplified to just resize for now as per main.py logic)
        # main.py simply resizes to input_size
        img = cv2.resize(img0, INPUT_SIZE)
    else:
        img = img0.copy()
        
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # HWC -> CHW
    img = np.expand_dims(img, axis=0) # Add batch dim
    return img0, Tensor(img, ms.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_outputs(outputs, anchors, input_shape):
    # outputs is a list of tensors from the user's simplified YOLOv5 model
    # Model has 1 detection head as per yolo.py source
    
    # output[0] shape: (Batch, NumAnchors * (5+NC), GridH, GridW)
    # e.g. (1, 3*(5+3), 20, 20) with stride 32 if 640 input
    
    det_out = outputs[0].asnumpy()
    
    bs, _, ny, nx = det_out.shape
    na = len(anchors)
    no = det_out.shape[1] // na # 5 + nc
    nc = no - 5
    
    # Reshape: (B, NA, NO, NY, NX) -> (B, NA, NY, NX, NO)
    det_out = det_out.reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
    
    # Make grid
    yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    grid = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)
    
    # Process
    # Sigmoid on xy, obj, cls
    pred = sigmoid(det_out) # (..., NO)
    
    # Decode xy
    # xy = (sigmoid(xy) * 2 - 0.5 + grid) * stride  <-- if using scaled YOLO
    # But yolo.py shows standard convs.. let's assume standard YOLOv5 box decoding
    # Standard YOLOv5: xy = (sigmoid(xy) * 2 - 0.5 + grid) * stride (usually)
    # Simplified: xy = (sigmoid(xy) + grid) / stride (if old yolo)
    # The yolo.py anchors: [[10,13, 16,30, 33,23]] -> these look like pixel values.
    # Stride is input / grid = 640 / 32 = 20? 
    # Wait, yolo.py says 3 downsamples (stride 8) then repeats.. 
    # yolo.py trace:
    # Conv(s=2) -> P1/2
    # Conv(s=2) -> P2/4
    # Conv(s=2) -> P3/8
    # Conv(s=2) -> P4/16
    # Conv(s=2) -> P5/32
    # So stride is 32.
    
    stride = input_shape[0] / ny
    anchor_grid = np.array(anchors).reshape(1, na, 1, 1, 2).astype(np.float32)
    
    # Decoding:
    # x, y = (sigmoid(tx) * 2 - 0.5 + cx) * stride
    # w, h = (sigmoid(tw) * 2) ** 2 * anchors
    # However, let's stick to simple decoding if uncertain:
    # box values are roughly 0-1 relative to grid cell?
    # Let's try standard YOLOv5 decoding:
    
    pred[..., 0:2] = (pred[..., 0:2] * 2. - 0.5 + grid) * stride
    pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * anchor_grid
    
    # Convert xywh to xyxy
    pred_boxes = np.zeros_like(pred[..., :4])
    x, y, w, h = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    pred_boxes[..., 0] = x - w / 2
    pred_boxes[..., 1] = y - h / 2
    pred_boxes[..., 2] = x + w / 2
    pred_boxes[..., 3] = y + h / 2
    
    # Flatten
    # (B, NA, NY, NX, NO) -> (B, -1, NO)
    pred_flat = pred.reshape(bs, -1, no)
    boxes_flat = pred_boxes.reshape(bs, -1, 4)
    
    return np.concatenate((boxes_flat, pred_flat[..., 4:]), axis=-1)

def nms(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    prediction: (N, 5+C) [x1, y1, x2, y2, obj_conf, cls1_prob, cls2_prob...]
    """
    output = []
    
    for pi in prediction: # Iterate over batch
        # Filter by confidence (obj_conf * cls_conf)
        # Simplify: just filter by obj_conf > conf_thres
        pi = pi[pi[:, 4] > conf_thres]
        
        if len(pi) == 0:
            continue
            
        # Get best class
        # scores = pi[:, 5:] * pi[:, 4:5]
        # For simplicity, let's just use obj_conf as score and argmax for class
        cls_scores = pi[:, 5:]
        cls_ids = np.argmax(cls_scores, axis=1)
        # scores = cls_scores[np.arange(len(cls_scores)), cls_ids] * pi[:, 4]
        # Or usually in YOLOv5: conf = obj_conf * cls_conf
        conf = pi[:, 4] # Simplified
        
        boxes = pi[:, :4]
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), conf_thres, iou_thres)
        
        result = []
        if len(indices) > 0:
            for i in indices.flatten():
                result.append(np.concatenate((boxes[i], [conf[i]], [cls_ids[i]])))
        output.append(result)
        
    return output

def main():
    print("Setting up MindSpore context...")
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    
    print(f"Loading model from {MODEL_PATH}...")
    net = YOLOv5(nc=3)
    try:
        param_dict = load_checkpoint(MODEL_PATH)
        load_param_into_net(net, param_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Return for now if model missing to avoid confusing user, or proceed if mocking
        # Assuming model exists as seen in ls
    
    print(f"Processing image {IMAGE_PATH}...")
    img_orig, img_input = preprocess(IMAGE_PATH)
    
    print("Running inference...")
    net.set_train(False)
    outputs = net(img_input)
    
    print("Post-processing...")
    anchors = [[10,13], [16,30], [33,23]] # From yolo.py
    # Note: flatten anchors for the decoder function if it expects pairs or list of tuples
    # yolo.py has `anchors=[[10,13, 16,30, 33,23]]` (list of list of ints)
    # The decoder expects list of [w, h] pairs
    anchors_pairs = [[10,13], [16,30], [33,23]]
    
    decoded = decode_outputs(outputs, anchors_pairs, INPUT_SIZE)
    detections = nms(decoded, CONF_THRES, IOU_THRES)
    
    car_count = 0
    
    if len(detections) > 0:
        det = detections[0] # Single image
        print(f"Found {len(det)} objects.")
        
        for d in det:
            x1, y1, x2, y2, conf, cls = d
            label = CLASSES[int(cls)]
            
            # Rescale to original image
            h_orig, w_orig = img_orig.shape[:2]
            scale_x = w_orig / INPUT_SIZE[1]
            scale_y = h_orig / INPUT_SIZE[0]
            
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            
            if label == 'Car':
                car_count += 1
                color = (0, 255, 0)
            elif label == 'Truck':
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
                
            cv2.rectangle(img_orig, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img_orig, f"{label} {conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    print(f"Total cars detected: {car_count}")
    
    output_path = "detected_cars.jpg"
    cv2.imwrite(output_path, img_orig)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()
