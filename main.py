from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from food import get_dishes_by_mood , get_dishes_by_region 
import json
 
app = FastAPI()

origins = [
    "http://localhost:3000",  # Frontend origin
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows requests from listed origins
    allow_credentials=True,
    allow_methods=["*"],    # Allows all HTTP methods
    allow_headers=["*"],    # Allows all headers
)

class PromptRequest(BaseModel):
    prompt: str
  
@app.get("/")
async def root():
    return {"message": "Hello! Welcome to Food/Mood"}

@app.post("/region")
def region(request: PromptRequest):        
    result = get_dishes_by_region(str(request))
    return result

@app.post("/mood")
def mood(request: PromptRequest):    
    result = get_dishes_by_mood(str(request))
    return result

 