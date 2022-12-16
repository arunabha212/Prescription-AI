import json
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from dta import dta
import build_model
from build_model import extract_drug_entity
import cv2
import pytesseract
# 2. Create the app object
app = FastAPI()


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Fast API': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted value
@app.post('/data')
def drug_predict(text:dta):
        # print(text)
        text=text.dict()
        txt=text['txt']
        print(type(txt))
        prediction = extract_drug_entity(txt)
        print(prediction)
        return  prediction


@app.post('/OCR')
def img_to_speech():
    pytesseract.pytesseract.tesseract_cmd = "E:\TesseractOCR\Tesseract.exe"

    video = cv2.VideoCapture("https://192.168.137.217:8080/video")

    video.set(3, 640)
    video.set(4, 480)

    extra, frames = video.read()
    data4 = pytesseract.image_to_data(frames)
    #print(data4.splitlines())
    str = ""
    for z, a in enumerate(data4.splitlines()):
        # Counter
        if z != 0:
            # Converts 'data1' string into a list stored in 'a'
            a = a.split()
            # Checking if array contains a word
            if len(a) == 12:
                # Storing values in the right variables
                x, y = int(a[6]), int(a[7])
                w, h = int(a[8]), int(a[9])
                # Display bounding box of each word
                cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Display detected word under each bounding box
                cv2.putText(frames, a[11], (x - 15, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
                str = str + a[11] + " "
    # show_width, show_height = 1500, 1000
    cv2.imshow('Image output', frames)
    cv2.waitKey(0)
    return str
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
   

