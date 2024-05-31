import torch
import cv2
import pyttsx3
import requests
import json
import numpy as np
import time
import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openai

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Assuming you're using a pre-trained YOLOv5 model

engine = pyttsx3.init()

openai.api_key = '<YOUR API KEY>' #Create an account on OpenAI and get the API key

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to get the details of the object from Wikipedia
def get_object_details(object_name):
    try:
        url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{object_name}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', f"Information about {object_name} is not available")
        else:
            return f"Information about {object_name} is not available"
    except requests.RequestException as e:
        return f"Information about {object_name} is not available"

# Function to get the position of the object in the frame
def get_position(bbox, frame_width):
    center_x = (bbox[0] + bbox[2]) / 2
    if center_x < frame_width / 3:
        return "left"
    elif center_x < 2 * frame_width / 3:
        return "center"
    else:
        return "right"

# Function to process the frames and detect objects
def process_frames(frame):
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    frame_height, frame_width, _ = frame.shape

    detected_objects = []

    for i, (label, bbox) in enumerate(zip(labels, cord)):
        class_name = model.names[int(label)]
        x1, y1, x2, y2 = int(bbox[0] * frame_width), int(bbox[1] * frame_height), int(bbox[2] * frame_width), int(
            bbox[3] * frame_height)
        position = get_position([x1, y1, x2, y2], frame_width)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {bbox[4]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        speak(f"I see a {class_name} on the {position}.")

        details = get_object_details(class_name)
        speak(details)

        detected_objects.append({"class_name": class_name, "position": position, "details": details})

        time.sleep(2)

    return frame, detected_objects

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"Recognized: {command}")
        return command.lower()
    except sr.UnknownValueError:
        return "Sorry, I didn't get that. Please try again."
    except sr.RequestError as e:
        return f"Could not request results; {e}"

from transformers import pipeline

# Explicitly specify the model name and revision
qa_pipeline = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad", revision="626af31")

def ask_openai(question, context=""):
    try:
        response = qa_pipeline(question=question, context=context)
        answer = response['answer']
        return answer
    except Exception as e:
        return f"An error occurred while getting the response: {str(e)}"

# Function to handle the command
def handle_command(command, detected_objects):
    context = " ".join([obj['details'] for obj in detected_objects])
    if "repeat" in command:
        speak("I will repeat the information.")
        for obj in detected_objects:
            speak(f"I see a {obj['class_name']} on the {obj['position']}. {obj['details']}")
    elif "details" in command:
        for obj in detected_objects:
            speak(f"Here are more details about {obj['class_name']}. {obj['details']}")
    else:
        answer = ask_openai(command, context)
        speak(answer)
        speak("Do you need any further assistance?")

# Main function to process the frames and handle the commands
def main(source_type="camera"):
    source = 0 if source_type == "camera" else source_type
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open the camera {source}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_objects = process_frames(frame)
        cv2.imshow("YOLO Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        command = recognize_speech()
        if command:
            handle_command(command, detected_objects)

    cap.release()
    cv2.destroyAllWindows()

# Entry point of the program
if __name__ == '__main__':
    source_type = input("Please enter either camera or file: ").strip()
    if source_type == "file":
        file_path = input("Please enter the file path: ").strip()
        main(source_type=file_path)
    else:
        main(source_type="camera")