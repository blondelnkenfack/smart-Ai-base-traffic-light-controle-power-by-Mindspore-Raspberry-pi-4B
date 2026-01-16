import cv2
import numpy as np
import time
import argparse
import os
from rpi_deployment.inference_lite import InferenceEngine
from rpi_deployment.controller import TrafficController
from core.traffic_manager import TrafficManager

def preprocess_frame(frame, input_size=(640, 640)):
    # Resize and normalize for YOLO
    img = cv2.resize(frame, input_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # HWC -> CHW
    img = np.expand_dims(img, axis=0) # Add batch dim
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5_traffic.ckpt', help='Path to MindSpore model (Lite .mindir or Full .ckpt)')
    # Default to the user's provided video path
    default_video = r"C:\Users\SSD\Desktop\Design of a smart AI-Based Traffic light system using ESP32-CAM and Edgeimpulse\traffic1.mp4"
    parser.add_argument('--source', type=str, default=default_video, help='Path to video file or Camera ID')
    parser.add_argument('--lanes', type=int, default=4, help='Number of lanes to simulate/monitor')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI window')
    args = parser.parse_args()

    # 1. Initialize Components
    print("Initializing Smart Traffic System...")
    controller = TrafficController(num_lanes=args.lanes)
    manager = TrafficManager(num_lanes=args.lanes)
    engine = InferenceEngine(args.model)
    
    # Determine source (Camera ID or File Path)
    source = args.source
    if source.isdigit():
        source = int(source)
    else:
        # It's a file path
        if not os.path.exists(source):
            print(f"ERROR: Video file not found at: {source}")
            # Try to handle common path issues (e.g., quotes)
            source = source.strip('"').strip("'")
            if not os.path.exists(source):
                print("Exiting due to missing video source.")
                return

    print(f"Opening video source: {source}")
    # Force FFMPEG backend for video files to avoid GStreamer/C: confusion on Windows
    # If on Linux (as per your error trace), standard capture works too.
    if isinstance(source, str):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG if os.name == 'nt' else cv2.CAP_ANY) 
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        # Try one more fallback without backend specifier if string
        if isinstance(source, str):
             print("Retrying with default backend...")
             cap = cv2.VideoCapture(source)
             
        if not cap.isOpened():
             return

    print("System Running. Press 'q' to exit.")
    
    try:
        while True:
            # 2. Capture Frame
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame. Looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            height, width, _ = frame.shape
            
            # 3. Inference
            input_tensor = preprocess_frame(frame)
            outputs = engine.infer(input_tensor)
            
            # 4. Parse Detections
            detections = outputs[0] if len(outputs) > 0 else []
            
            # Define ROIs (Assuming overhead view)
            # 0: North, 1: East, 2: South, 3: West (Clockwise)
            # Use 1/3 splits for simplicity
            x_third = width / 3
            y_third = height / 3
            
            # Lane Counts
            lane_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            lane_names = {0: "North", 1: "East", 2: "South", 3: "West"}
            
            if len(detections) > 0 and len(detections.shape) > 1:
                # Resize scaling
                scale_x = width / 640.0
                scale_y = height / 640.0
                
                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det
                    
                    # Scale to original frame
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    # Determine Lane
                    lane_id = -1
                    if x_third < cx < 2*x_third and cy < y_third:
                        lane_id = 0 # North
                    elif cx > 2*x_third and y_third < cy < 2*y_third:
                        lane_id = 1 # East
                    elif x_third < cx < 2*x_third and cy > 2*y_third:
                        lane_id = 2 # South
                    elif cx < x_third and y_third < cy < 2*y_third:
                        lane_id = 3 # West
                        
                    if lane_id != -1:
                        lane_counts[lane_id] += 1
                        
                        # Draw Box
                        color = (0, 255, 0)
                        if cls_id == 1: color = (0, 255, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 5. Logic Decision
            # Construct simpler list for manager: [count_N, count_E, count_S, count_W]
            # Map simplified counts to the opaque "lane_data" expected by manager?
            # Manager expects {lane_idx: [cls, cls...]}
            # Let's mock the data structure so the manager logic still works
            lane_data_mock = {}
            for l_id, count in lane_counts.items():
                lane_data_mock[l_id] = [0] * count # Assume all cars for logic simplicity
                
            next_green_lane = manager.decide_next_state(lane_data_mock)
            
            # 6. Apply Control
            controller.set_green_lane(next_green_lane)
            
            # 7. Visualization
            # Draw dividers and info for each way
            
            # Stop Lines Coordinates
            # North Stop Line (Bottom of N zone)
            lines = {
                0: ((int(x_third), int(y_third)), (int(2*x_third), int(y_third))),
                1: ((int(2*x_third), int(y_third)), (int(2*x_third), int(2*y_third))),
                2: ((int(x_third), int(2*y_third)), (int(2*x_third), int(2*y_third))),
                3: ((int(x_third), int(y_third)), (int(x_third), int(2*y_third)))
            }
            
            # Text Positions
            text_pos = {
                0: (int(width/2) - 20, 30),
                1: (width - 100, int(height/2)),
                2: (int(width/2) - 20, height - 20),
                3: (10, int(height/2))
            }

            for i in range(4):
                is_green = (i == next_green_lane)
                color = (0, 255, 0) if is_green else (0, 0, 255)
                
                # Draw Stop Line
                pt1, pt2 = lines[i]
                cv2.line(frame, pt1, pt2, color, 4)
                
                # Draw Count
                label = f"{lane_names[i]}: {lane_counts[i]}"
                t_pos = text_pos[i]
                cv2.putText(frame, label, t_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Global Info
            cv2.putText(frame, f"Active Green: {lane_names[next_green_lane]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Visualize Quadrant Grid (Light Gray)
            cv2.line(frame, (int(x_third), 0), (int(x_third), height), (50, 50, 50), 1)
            cv2.line(frame, (int(2*x_third), 0), (int(2*x_third), height), (50, 50, 50), 1)
            cv2.line(frame, (0, int(y_third)), (width, int(y_third)), (50, 50, 50), 1)
            cv2.line(frame, (0, int(2*y_third)), (width, int(2*y_third)), (50, 50, 50), 1)

            # Save frame for checking
            cv2.imwrite('latest_view.jpg', frame)

            if not args.no_gui:
                try:
                    cv2.imshow('Smart Traffic Control', frame)
                    # Slow motion: wait 100ms instead of 1ms
                    if cv2.waitKey(2000) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"GUI Error: {e}. Switching to headless mode.")
                    args.no_gui = True
            else:
                 time.sleep(0.05)
                
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        controller.cleanup()

if __name__ == "__main__":
    main()
