import uvicorn
from fastapi import FastAPI, HTTPException
from knowledge_router import router

app = FastAPI()

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Knowledge based AI Bot application!"}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8002)