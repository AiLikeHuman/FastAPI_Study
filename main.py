from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import json
import io
import matplotlib.pyplot as plt
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


# Example endpoints from your previous code

@app.get('/')
def read_root():
    return {"Hello": "World"}

@app.get("/items/")
def read_item(q: Optional[str] = Query(default=None, max_length=50)):
    results = {"items": [{"item_id": "FOO"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results

# @app.put("/items/{item_id}")
# def update_item(
#     item_id: int, 
#     item: Item, 
#     user: User, 
#     importance: int = Body(gt=0), 
#     q: Optional[str] = None
# ):
#     results = {"item_price": item.price, "item_id": item_id, "user": user, "importance": importance}
#     if q:
#         results.update({"q": q})
#     return results

# @app.get("/models/{model_name}")
# def get_model(model_name: ModelName):
#     if model_name is ModelName.alexnet:
#         return {"model_name": model_name, "message": "Deep Learning"}
#     if model_name.value == 'lenet':
#         return {"model_name": model_name, "message": "LeCNN Learning"}

# @app.get("/items/{item_id}")
# async def read_user_item(item_id: str, needy: str, skip: int = 0, limit: Optional[int] = None):
#     item = {"item_id": item_id, "needy": needy, "skip": skip, "limit": limit}
#     return item

# @app.post("/items/")
# async def create_item(item: Item):
#     return item

# def get_name_with_age(name: str, age: int) -> str:
#     name_with_age = name + " is this old: " + str(age)
#     return name_with_age

# def process_items(items_t: Tuple[int, int, str], items_s: Set[bytes]) -> Tuple[Set[bytes], Tuple[int, int, str]]:
#     return items_s, items_t

# def process_items2(prices: Dict[str, float]):
#     for item_name, item_price in prices.items():
#         print(item_name)
#         print(item_price)

# def say_hi(name: Optional[str] = None):
#     if name is not None:
#         print(f"Hey {name}!")
#     else:
#         print("Hello World")

class Person:
    def __init__(self, name: str):
        self.name = name

def get_person_name(one_person: Person) -> str:
    return one_person.name

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
