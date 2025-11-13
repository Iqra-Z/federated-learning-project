import torch
from fl_core.model_def import get_model
from copy import deepcopy

class Aggregator:
    def __init__(self):
        self.global_model = get_model()
        self.current_round = 0
        self.updates = []
        self.data_sizes = []

    def get_weights(self):
        return {k: v.cpu().tolist() for k, v in self.global_model.state_dict().items()}

    def receive_update(self, weights, data_size):
        self.updates.append(weights)
        self.data_sizes.append(data_size)

    def aggregate(self):
        if len(self.updates) == 0:
            return

        new_state = deepcopy(self.global_model.state_dict())

        # Weighted averaging
        total_data = sum(self.data_sizes)

        for key in new_state.keys():
            new_state[key] = sum(
                (torch.tensor(update[key]) * (ds / total_data))
                for update, ds in zip(self.updates, self.data_sizes)
            )

        self.global_model.load_state_dict(new_state)
        self.current_round += 1
        self.updates = []
        self.data_sizes = []
