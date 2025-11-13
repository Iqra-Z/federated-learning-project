
from fastapi import FastAPI
from pydantic import BaseModel
from server.aggregator import Aggregator

app = FastAPI()
agg = Aggregator()

class UpdateRequest(BaseModel):
    weights: dict
    data_size: int

@app.get("/get_model")
def get_model():
    return {
        "round": agg.current_round,
        "weights": agg.get_weights()
    }

@app.post("/submit_update")
def submit_update(upd: UpdateRequest):
    agg.receive_update(upd.weights, upd.data_size)

    # aggregate once 3 clients send updates (change number if needed)
    if len(agg.updates) >= 3:
        agg.aggregate()

    return {"status": "received"}

@app.get("/metrics")
def metrics():
    return {
        "round": agg.current_round
    }
