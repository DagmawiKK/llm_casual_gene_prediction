from fastapi import FastAPI
from app.routes import inference

app = FastAPI()
app.include_router(inference.router)
