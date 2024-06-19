from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from interact import interact, get_history

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
async def answer(question: str, temperature: float = 1,  max_history_len: int = 10, max_len: int = 100,
                 repetition_penalty: float = 1):
    return interact(question, "./outputs/min_ppl_model/", max_history_len, max_len, repetition_penalty, temperature)

@app.get("/history")
async def history():
    return get_history()
