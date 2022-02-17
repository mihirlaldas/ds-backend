from fastapi import FastAPI, File, UploadFile, Form
from sklearn.model_selection import train_test_split
import pandas as pd

app = FastAPI()


@app.get("/")
def index():
    return {"name": "First Data"}


@app.post("/uploadfile/")
async def create_upload_file(response: str = Form(...), algo: str = Form(...), stratify_col: str = Form(default=''),
                             csv_file: UploadFile = File(...)):
    vehicles = pd.read_csv(csv_file.file)
    # response = 'co2emissions'
    y = vehicles[[response]]
    predictors = list(vehicles.columns)
    predictors.remove(response)
    x = vehicles[predictors]
    if algo == "simple":
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, stratify=x[stratify_col])
    x_train.to_csv('x_train.csv', encoding='utf-8')
    x_test.to_csv('x_test.csv', encoding='utf-8')

    y_train.to_csv('y_train.csv', encoding='utf-8')
    y_test.to_csv('y_test.csv', encoding='utf-8')
    return {"message": " samples created: x_train, x_test, y_train, y_test"}
