from fastapi import FastAPI, File, UploadFile, Form
from sklearn.model_selection import train_test_split
import pandas as pd

app = FastAPI()


@app.get("/")
def index():
    return {"name": "First Data"}


@app.post("/randomsample/")
async def generate_samples(samples: int = Form(default=1), ratio: float = Form(default=0.75), csv_file: UploadFile = File(...)):
    print(samples, csv_file)
    population = pd.read_csv(csv_file.file)
    train = population.sample(frac=ratio)
    train.to_csv('1_train.csv', encoding='utf-8')
    remaining_population = population.loc[~population.index.isin(train.index), :]
    remaining_samples = samples-1
    for i in range(1, remaining_samples + 1):
        test = remaining_population
        test = test.sample(frac=i/remaining_samples)
        test.to_csv(f'{i}_test.csv', encoding='utf-8')
        remaining_population = remaining_population.loc[~remaining_population.index.isin(test.index), :]
    return {"message": f'{samples} sample created'}


@app.post("/sklearnsample/")
async def create_upload_file(response: str = Form(...), algo: str = Form(...), stratify_col: str = Form(default=''),
                             csv_file: UploadFile = File(...)):
    population = pd.read_csv(csv_file.file)
    # response = 'co2emissions'
    y = population[[response]]
    predictors = list(population.columns)
    predictors.remove(response)
    x = population[predictors]
    if algo == "simple":
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, stratify=x[stratify_col])
    x_train.to_csv('x_train.csv', encoding='utf-8')
    x_test.to_csv('x_test.csv', encoding='utf-8')

    y_train.to_csv('y_train.csv', encoding='utf-8')
    y_test.to_csv('y_test.csv', encoding='utf-8')
    return {"message": " samples created: x_train, x_test, y_train, y_test"}
