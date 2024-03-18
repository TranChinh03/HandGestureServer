from fastapi import APIRouter, FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
from app import session

router = APIRouter(prefix="/predict", tags=["Predict"])

classes = ["call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "three2",
    "two_up",
    "two_up_inverted",
    "no_gesture"];

async def process_image(file):
    # Open the image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Resize the image to desired dimensions (640x640 in this case)
    img = img.resize((640, 640))
    
    # Convert the image to RGB (if not already in RGB mode)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Normalize the pixel values to be in the range [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    # Transpose the array to match the shape [1, 3, 640, 640]
    img_array = np.transpose(img_array, (2, 0, 1))  # Change channel order to be first
    
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

async def postprocess(output):

    # Transfer elements
    reshaped_array = np.transpose(output, (0, 2, 1))

    detections = []
    for detection in reshaped_array[0]:
        box = detection[:4].tolist()
        class_confidence = detection[4:].tolist()
        max_confidence = max(class_confidence)
        max_index = class_confidence.index(max_confidence)
        detections.append({"box": box, "confidence": max_confidence, "class_id": max_index, "class_name": classes[max_index]})
        # Sort the data based on confidence in descending order
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        detections = detections[0:10]

    return detections

@router.post("")
async def upload_image(file: UploadFile = File(...)):
    handledImage = await process_image(file)
    if session is None:
        return "Kh load"
    inputs = session.get_inputs()
    for i, input_info in enumerate(inputs):
        print(f"Input {i}: name = {input_info.name}, shape = {input_info.shape}")
    input_name = session.get_inputs()[0].name

    outputs = session.get_outputs()
    for i, output_info in enumerate(outputs):
        print(f"Output {i}: name = {output_info.name}, shape = {output_info.shape}")
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: handledImage})[0]

    detections = await postprocess(prediction)

    # Dictionary to keep track of highest confidence for each class_id
    highest_confidence = {}
    # Iterate through the data to find the highest confidence for each class_id 
    for entry in detections:
        class_id = entry['class_id']
        confidence = entry['confidence']
        if class_id not in highest_confidence or confidence > highest_confidence[class_id]['confidence']:
            highest_confidence[class_id] = entry

    # Filter out duplicates based on highest confidence for each class_id
    filtered_data = list(highest_confidence.values())

    return filtered_data