# Smart AI Traffic Light Control (MindSpore)

This project implements a smart traffic light control system using the **MindSpore** framework on a **Raspberry Pi 4B**. It uses **YOLOv5** to detect traffic density and dynamically adjust traffic lights to reduce waiting time and prioritize emergency vehicles.

## Features
*   **Object Detection**: Detects `Car`, `Truck`, and `Emergency` vehicles using MindSpore.
*   **Smart Priority Logic**: Calculates lane density. High density = Green light.
*   **Heavy Vehicle Weighting**: Trucks count as 2x standard cars.
*   **Emergency Override**: Emergency vehicles get immediate priority.
*   **Starvation Prevention**: Ensures no lane waits indefinitely.

## Hardware Requirements
*   Raspberry Pi 4B (4GB+ RAM recommended)
*  we used Pi Camera Module 
*   Traffic Light LED Modules (x4 lanes)
*   Jumper Wires

## Wiring Guide (GPIO BCM)

| Lane | Red | Yellow | Green |
|------|-----|--------|-------|
| 1    | 17  | 27     | 22    |
| 2    | 10  | 9      | 11    |
| 3    | 5   | 6      | 13    |
| 4    | 19  | 26     | 21    |

## Installation

1.  **Install Dependencies**:
    ```bash
    pip install numpy opencv-python RPi.GPIO
    ```
    *Note: MindSpore Lite must be installed separately for the specific Pi OS version.*

2.  **Run the System**:
    ```bash
    python main.py --model traffic_model.mindir --lanes 4
    ```

3.  **Simulation Mode**:
    If no camera is connected, the system will run with black frames (check `main.py` to enable mock detections for testing).

## Development
*   `model/`: Contains MindSpore YOLOv5 definitions.
*   `train.py`: Script to train the model on my PC.
*   `core/traffic_manager.py`: The logic brain for traffic decisions.
