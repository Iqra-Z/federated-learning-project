import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import torch
from fl_core.model_def import get_model
from clients.data_utils import get_dataloader
from clients.training import train_local

SERVER_URL = "http://127.0.0.1:9000"

def client_loop(client_id):
    train_loader, _ = get_dataloader()

    while True:
        print(f"[Client {client_id}] Fetching global model...")
        r = requests.get(f"{SERVER_URL}/get_model").json()
        global_round = r["round"]

        # Load server weights
        model = get_model()
        state = {k: torch.tensor(v) for k, v in r["weights"].items()}
        model.load_state_dict(state)

        print(f"[Client {client_id}] Training locally...")
        model = train_local(model, train_loader, epochs=1)

        weights_to_send = {k: v.cpu().tolist() for k, v in model.state_dict().items()}

        print(f"[Client {client_id}] Sending update...")
        requests.post(f"{SERVER_URL}/submit_update", json={
            "weights": weights_to_send,
            "data_size": len(train_loader.dataset)
        })

        print(f"[Client {client_id}] Update sent. Waiting next round...\n")
        import time
        time.sleep(3)


if __name__ == "__main__":
    import sys
    client_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    client_loop(client_id)
