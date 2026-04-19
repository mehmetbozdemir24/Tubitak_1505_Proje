from fastapi import FastAPI

app = FastAPI()

@app.get("/home")
def read_home():
    return {"message": "merhaba dünya!"}