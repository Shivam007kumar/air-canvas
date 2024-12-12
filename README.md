# Hand Gesture Controlled Whiteboard

This project is a hand gesture-controlled digital whiteboard using OpenCV and MediaPipe. Users can draw, erase, and interact with a whiteboard through webcam-detected hand movements.

## Features
- Hand gesture-controlled drawing and erasing.
- Dynamic color palette and brush size selection.
- Undo functionality.
- Line mode and clear canvas options.
- Save drawings as images.

## Requirements
- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/username/whiteboard-drawing.git
    cd whiteboard-drawing
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    python main.py
    ```

## Controls
- **Draw**: Move your hand.
- **Change Color/Brush Size**: Hover over the palette/size buttons.
- **Erase**: Select the eraser button.
- **Clear Canvas**: Click the "Clear" button.
- **Undo**: Press `U`.
- **Save Drawing**: Press `S`.
- **Toggle Line Mode**: Press `L`.
- **Pause/Resume Drawing**: Press `Space`.
- **Exit**: Press `Q`.

## Example Output
![Sample Output](examples/sample_output.png)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
