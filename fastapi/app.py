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

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
   

