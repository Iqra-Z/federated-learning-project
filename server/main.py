"""
Federated Learning Server - Complete Implementation
===================================================

This server coordinates federated learning between multiple edge clients.

ARCHITECTURE:
- Clients train locally on their data
- Clients send model updates (not raw data) to this server
- Server aggregates updates using Federated Averaging (FedAvg)
- Server evaluates global model accuracy
- Dashboard connects to monitor training in real-time

ENDPOINTS:
- GET  /                 : Health check
- GET  /get_model        : Clients download global model
- POST /submit_update    : Clients submit trained updates
- GET  /status          : Dashboard gets system status
- GET  /metrics         : Dashboard gets training metrics
"""

import os
import sys
from datetime import datetime

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path so we can import from server package
# This allows the script to find 'server.aggregator' when run from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.aggregator import Aggregator

# ============================================================
# INITIALIZE FASTAPI APP
# ============================================================
app = FastAPI(
    title="Federated Learning Server",
    description="Coordinates distributed machine learning across edge devices",
    version="1.0.0",
)

# ============================================================
# CORS MIDDLEWARE - Critical for Dashboard Connection
# ============================================================
# WHAT IS CORS?
# Cross-Origin Resource Sharing allows your React app (running on
# localhost:3000) to make requests to this server (localhost:9000).
# Without this, browsers block the requests for security.
#
# ANALOGY: Like a security guard allowing visitors from building 3000
# to enter building 9000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (fine for development)
    allow_credentials=True,  # Allow cookies/auth headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ============================================================
# GLOBAL STATE
# ============================================================
# Create ONE aggregator instance that persists across requests
# Why global? All clients need to share the same model state
agg = Aggregator()

# Track client activity for dashboard visualization
# Structure: {client_id: {status, timestamp, data_size, round}}
clients_status = {}

# Store training history for charts
# Structure: [{round, accuracy, timestamp, loss}, ...]
training_history = []

# Configuration
UPDATES_PER_ROUND = 3  # Wait for this many client updates before aggregating


# ============================================================
# DATA MODELS (Type Validation)
# ============================================================
# Pydantic models validate incoming data automatically
# If a client sends wrong format, FastAPI returns error
class UpdateRequest(BaseModel):
    """Data structure for client updates"""

    weights: dict  # Model weights as nested dict of lists
    data_size: int  # Number of training samples client used

    class Config:
        # Example for API documentation
        schema_extra = {
            "example": {
                "weights": {"layer1.weight": [[0.1, 0.2], [0.3, 0.4]]},
                "data_size": 6000,
            }
        }


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/")
def root():
    """
    Health check endpoint - confirms server is running

    USE CASE: Monitoring, debugging, automated health checks

    EXAMPLE RESPONSE:
    {
        "status": "Federated Learning Server Running",
        "round": 5,
        "clients_seen": 10,
        "pending_updates": 2
    }
    """
    return {
        "status": "Federated Learning Server Running",
        "round": agg.current_round,
        "clients_seen": len(clients_status),
        "pending_updates": len(agg.updates),
        "version": "1.0.0",
    }


@app.get("/get_model")
def get_model():
    """
    Clients download the current global model

    WORKFLOW:
    1. Client requests model weights
    2. Server converts PyTorch tensors → Python lists (JSON compatible)
    3. Client receives weights and loads them into local model

    WHY RETURN LISTS?
    PyTorch tensors can't be sent over HTTP (not JSON serializable).
    We convert to lists for transmission, client converts back to tensors.

    EXAMPLE RESPONSE:
    {
        "round": 5,
        "weights": {
            "conv1.weight": [[0.1, 0.2], ...],
            "conv1.bias": [0.3, 0.4, ...],
            ...
        }
    }
    """
    return {
        "round": agg.current_round,
        "weights": agg.get_weights(),  # Already returns dict of lists
    }


@app.post("/submit_update")
def submit_update(upd: UpdateRequest):
    """
    Clients submit their trained model updates

    WORKFLOW:
    1. Client trains locally (never sends raw data)
    2. Client sends model update (difference from global model)
    3. Server stores update
    4. If enough updates (3), server aggregates them
    5. Server evaluates new global model
    6. Server stores metrics for dashboard

    WHY WAIT FOR 3 UPDATES?
    - Federated learning uses "rounds" with multiple clients
    - Aggregating too often: inefficient, noisy updates
    - Aggregating too rarely: slow convergence
    - 3 is a balance for demo purposes (adjust UPDATES_PER_ROUND)

    EXAMPLE REQUEST:
    {
        "weights": {"layer1.weight": [[0.1, 0.2], ...]},
        "data_size": 6000
    }

    EXAMPLE RESPONSE:
    {
        "status": "received",
        "round": 5,
        "pending_updates": 1,
        "message": "Update stored. Waiting for 2 more updates."
    }
    """
    # Store the update in aggregator
    agg.receive_update(upd.weights, upd.data_size)

    # Track this client for dashboard
    client_id = len(clients_status)  # Simple ID: 0, 1, 2, ...
    clients_status[client_id] = {
        "status": "submitted",
        "timestamp": datetime.now().isoformat(),
        "data_size": upd.data_size,
        "round": agg.current_round,
    }

    print(
        f"[SERVER] ✓ Client {client_id} submitted update. "
        f"Updates: {len(agg.updates)}/{UPDATES_PER_ROUND}"
    )

    # Check if we have enough updates to aggregate
    if len(agg.updates) >= UPDATES_PER_ROUND:
        print(f"[SERVER] 🔄 Aggregating round {agg.current_round + 1}...")

        # Perform Federated Averaging
        agg.aggregate()

        # Evaluate the new global model
        accuracy, loss = evaluate_model()

        # Store metrics for dashboard charts
        training_history.append(
            {
                "round": agg.current_round,
                "accuracy": accuracy,
                "loss": loss,
                "timestamp": datetime.now().isoformat(),
                "num_clients": UPDATES_PER_ROUND,
            }
        )

        print(
            f"[SERVER] ✅ Round {agg.current_round} complete. "
            f"Accuracy: {accuracy:.2%}, Loss: {loss:.4f}"
        )

        return {
            "status": "aggregated",
            "round": agg.current_round,
            "accuracy": accuracy,
            "message": f"Round {agg.current_round} complete!",
        }

    # Not enough updates yet
    remaining = UPDATES_PER_ROUND - len(agg.updates)
    return {
        "status": "received",
        "round": agg.current_round,
        "pending_updates": len(agg.updates),
        "message": f"Update stored. Waiting for {remaining} more updates.",
    }


@app.get("/status")
def get_status():
    """
    Dashboard calls this to get current system state

    PURPOSE: Provide real-time snapshot of the entire system

    CALLED BY: Dashboard every 2 seconds via polling

    RETURNS:
    - current_round: Which training round we're on
    - active_clients: How many clients have submitted
    - pending_updates: Updates waiting to be aggregated
    - clients: Details about each client
    - history: Last 20 rounds of training metrics

    EXAMPLE RESPONSE:
    {
        "current_round": 5,
        "active_clients": 10,
        "pending_updates": 2,
        "clients": {
            "0": {"status": "submitted", "data_size": 6000, ...},
            "1": {"status": "submitted", "data_size": 5800, ...}
        },
        "history": [
            {"round": 1, "accuracy": 0.82, "loss": 0.45},
            {"round": 2, "accuracy": 0.85, "loss": 0.38},
            ...
        ]
    }
    """
    return {
        "current_round": agg.current_round,
        "active_clients": len(clients_status),
        "pending_updates": len(agg.updates),
        "clients": clients_status,
        "history": training_history[-20:],  # Last 20 rounds only
    }


@app.get("/metrics")
def get_metrics():
    """
    Dashboard calls this to get training metrics for charts

    PURPOSE: Provide data formatted for visualization

    WHY SEPARATE FROM /status?
    - Status changes every 2 seconds (client activity)
    - Metrics only change after aggregation (less frequent)
    - Separating reduces unnecessary data transfer

    CHART FRIENDLY FORMAT:
    Returns parallel arrays for easy plotting:
    - rounds: [1, 2, 3, 4, 5]
    - accuracy: [0.82, 0.85, 0.87, 0.89, 0.91]
    - loss: [0.45, 0.38, 0.32, 0.28, 0.24]

    EXAMPLE RESPONSE:
    {
        "round": 5,
        "rounds": [1, 2, 3, 4, 5],
        "accuracy": [0.82, 0.85, 0.87, 0.89, 0.91],
        "loss": [0.45, 0.38, 0.32, 0.28, 0.24],
        "total_history": 5
    }
    """
    return {
        "round": agg.current_round,
        "rounds": [h["round"] for h in training_history],
        "accuracy": [h["accuracy"] for h in training_history],
        "loss": [h["loss"] for h in training_history],
        "total_history": len(training_history),
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def evaluate_model():
    """
    Evaluate the global model on test data

    WHAT IS EVALUATION?
    Testing the model on data it has NEVER seen during training.
    This measures real-world performance.

    ANALOGY:
    - Training: Studying with practice problems
    - Evaluation: Taking the actual exam with new questions

    PROCESS:
    1. Load test dataset (10,000 MNIST images)
    2. Run each image through model
    3. Compare prediction to actual label
    4. Calculate accuracy and loss

    WHY model.eval()?
    - Disables dropout (regularization technique)
    - Disables batch normalization updates
    - Makes predictions deterministic

    WHY torch.no_grad()?
    - Disables gradient computation (saves memory)
    - We're not training, so don't need gradients
    - Makes evaluation much faster

    RETURNS:
    - accuracy: float (0.0 to 1.0) - % of correct predictions
    - loss: float - average cross-entropy loss
    """
    import torch.nn.functional as F

    from clients.data_utils import get_dataloader

    # Load test dataset
    _, test_loader = get_dataloader(batch_size=128)

    # Get model and set to evaluation mode
    model = agg.global_model
    model.eval()

    # Initialize counters
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    # Disable gradient computation
    with torch.no_grad():
        for images, labels in test_loader:
            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            # Get predictions (class with highest probability)
            _, predicted = torch.max(outputs.data, 1)

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate final metrics
    accuracy = correct / total
    avg_loss = total_loss / num_batches

    return accuracy, avg_loss


# ============================================================
# SERVER STARTUP
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("🚀 FEDERATED LEARNING SERVER STARTING")
    print("=" * 70)
    print()
    print("📍 Server URL:     http://127.0.0.1:9000")
    print("📊 API Docs:       http://127.0.0.1:9000/docs")
    print("🎨 Dashboard:      http://localhost:3000 (start React app)")
    print()
    print("⚙️  Configuration:")
    print(f"   - Aggregation:  Every {UPDATES_PER_ROUND} client updates")
    print(f"   - Dataset:      MNIST (handwritten digits)")
    print(f"   - Algorithm:    Federated Averaging (FedAvg)")
    print()
    print("🤖 Start clients:")
    print("   python clients/client.py 1 --num-clients 10 --non-iid")
    print("   python clients/client.py 2 --num-clients 10 --non-iid")
    print("   python clients/client.py 3 --num-clients 10 --non-iid")
    print()
    print("=" * 70)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=9000)
