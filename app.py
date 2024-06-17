from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def hello_world():  # put application's code here
    return 'Hello World!'

@app.get("/answer")
async def answer(question: str):
    return {"answer": question}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="app:app", host="localhost", port=8000, reload=True)