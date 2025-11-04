# main.py
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client
import asyncio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from frontend
origins = [
    "http://localhost:3000",  # React dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow requests from these origins
    allow_credentials=True,
    allow_methods=["*"],     # allow all HTTP methods
    allow_headers=["*"],     # allow all headers
)

# Hugging Face Space
HF_SPACE = "Andrewwinn/senior-project-demo"

class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    input: str
    prediction: dict  # Adjust based on what the model returns

@app.post("/analyze")
async def analyze(input_data: TextInput):
    """
    Analyze sentiment using Hugging Face Space
    """
    try:
        # Connect to Hugging Face Space
        client = Client(HF_SPACE)
        
        # Make prediction (run in thread to not block async)
        result = await asyncio.to_thread(
            client.predict,
            input_data.text,
            api_name="/predict"
        )
        
        return {
            "input": input_data.text,
            "prediction": result
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error calling Hugging Face Space: {str(e)}"
        )

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running"}