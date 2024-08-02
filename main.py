from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import json
import io
from datetime import datetime

# Create the FastAPI app
app = FastAPI()

# Load your trained model
model = EfficientNet.from_name('efficientnet-b0', num_classes=4)
checkpoint = torch.load('weights/best_weights_b0_card_epoch_10.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

# Load ImageNet class names
with open('labels_card.txt') as f:
    labels_map = json.load(f)

num_classes = len(labels_map)
labels_map = [labels_map[str(i)] for i in range(num_classes)]

# Define the image transformations
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        
        # Preprocess the image
        img = tfms(image).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            outputs = model(img)

        # Get the top 4 predictions
        topk_indices = torch.topk(outputs, k=4).indices.squeeze(0).tolist()

        predictions = []
        for idx in topk_indices:
            if idx < num_classes:
                prob = torch.softmax(outputs, dim=1)[0, idx].item()
                predictions.append({'label': labels_map[idx], 'probability': prob * 100})
            else:
                predictions.append({'label': f'Class index {idx} out of range', 'probability': 0})

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
def read_root():
    return {"Hello": "World"}

@app.get("/items/")
def read_item(q: Optional[str] = None):
    results = {"items": [{"item_id": "FOO"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results

class UserPydantic(BaseModel):
    id: int
    name: str = "John Doe"
    signup_ts: Optional[datetime] = None
    friends: List[int] = []

external_data = {
    "id": "123",
    "signup_ts": "2017-06-01 12:22",
    "friends": [1, "2", b"3"],
}

user_pydantic = UserPydantic(**external_data)
