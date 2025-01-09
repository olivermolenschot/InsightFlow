from fastapi import FastAPI, UploadFile, Form 
from .train import dataset_to_model
import shutil 

app = FastAPI()

@app.post("/train")
async def train_endpoint(
    dataset: UploadFile,
    epochs: int = Form(10),
    learning_rate: float = Form(0.01),
    batch_size: int = 1,
    ):
    """
    API endpoint to train a model on the uploaded dataset
    """
    dataset_path = f"./{dataset.filename}"
    with open(dataset_path, "wb") as buffer:
        shutil.copyfileobj(dataset.file, buffer)

    trainer = dataset_to_model(batch_size=batch_size,data_path=dataset_path,learning_rate=learning_rate,num_epochs=epochs)
    model_path = trainer.path

    return {
        "message": "Trainer Complete",
        "model_path": model_path,
    }
