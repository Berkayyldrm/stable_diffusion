from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from app.controllers import controller

app = FastAPI()

app.include_router(controller.router, prefix="/ml", tags=["ml"])
app.mount("/static", StaticFiles(directory="static"), name="static")
if __name__ == "__main__":
    uvicorn.run("main_app:app", host="0.0.0.0", port=8000, reload=True)#, reload=True
