from fastapi import FastAPI, UploadFile, Form 
import shutil 
import os
import torch 
import torch.nn as nn
from .model import Classifier
from .data_processing import MyDataset

app = FastAPI()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model file
model_path = os.path.join(script_dir, 'model_weights.pth')

if not os.path.isfile(model_path):
    raise RuntimeError('Model file not found, you must train a model beforehand using the train command.')

model = Classifier()
model.load_state_dict(torch.load(model_path))
model.eval()
model.to('cpu')

@app.post('/inference')
async def inference_endpoint(dataset: UploadFile):
    """
    API endpoint to run inference on the model trained
    """
    saved_file_path = f"./uploaded_files/{dataset.filename}"
    os.makedirs(os.path.dirname(saved_file_path), exist_ok=True)  
    with open(saved_file_path, "wb") as buffer:
        shutil.copyfileobj(dataset.file, buffer)

    dataset = MyDataset(csv_path=saved_file_path)

    # Run inference
    output = []
    for inputs,labels in dataset:
        prediction = model(inputs).item()  # Assuming model outputs a single value
        output.append(prediction)

    return {"predictions": output}

