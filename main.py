from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
from datetime import datetime

# Create the FastAPI app
app = FastAPI()


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
