# Create a fastapi app to for the user to input the data and get the prediction also can train the model using the app
import os
import streamlit as st
from src.textSummarizer.pipeline.prediction import PredictionPipeline
from src.textSummarizer.pipeline.stage02_model_trainer import ModelTrainingPipeline
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

text:str = "Check what is the category of the text is"

app = FastAPI()

@app.get("/")
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Model trained successfully")
    
    except Exception as e:
        return Response(f"Failed to train the model with exception: {e}")
 
prediction_label = {0: 'This Sentence is not offensive', 1: 'This Sentence is a Targeted Insult', 2: 'This Sentence is not a Targeted Insult, but is offensive'}
    
@app.get("/predict")
async def predict(text):
    try:
        pipeline = PredictionPipeline()
        result = pipeline.predict(text)
        result = prediction_label[result]

        return Response(result)
    
    except Exception as e:
        return Response(f"Failed to predict the category with exception: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


# st.title("Text Classification App")
# st.write("This is a simple text classification app")

# text = st.text_area("Enter the text to classify", "Type Here")

# if st.button("Classify"):
#     result = requests.get(f"http://0.0.0.0:8080/predict?text={text}")