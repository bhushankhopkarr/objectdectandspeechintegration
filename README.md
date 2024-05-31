# Object Detection and Speech Recognition

This project combines object detection using YOLOv5, speech recognition, and natural language processing (NLP) to create an interactive system for real-time object detection with speech-based interaction.

## Features

- **Object Detection**: Utilizes YOLOv5 model for real-time object detection.
- **Speech Recognition**: Interacts with users through speech commands.
- **Information Retrieval**: Retrieves detailed information about detected objects from Wikipedia using NLP.
- **Interactive Interface**: Provides real-time feedback and allows users to request additional information or repeat object details.

## Dependencies

- Torch
- OpenCV
- Pyttsx3
- SpeechRecognition
- NLTK
- OpenAI
- Transformers

## Usage

1. Run the script and specify the input source as either the camera or a video file.
2. The system detects objects in the frames and provides real-time feedback through speech synthesis.
3. Users can interact with the system by issuing voice commands such as requesting object details or asking questions about the detected objects.

## How to Run

```bash
python main.py
```
For file input:
```bash
python main.py file
```
## Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to help improve the project.

Replace `<YOUR API KEY>` with your actual OpenAI API key in the script and update the file paths accordingly before using the system.
