"""
FedSHA client: local training of the adapters in its windows, the gates, scale / shift and the mask decoder.
During training it records the gate mass of every adapter (mean gate probability over the round's tokens and the
L blocks; 0 outside its windows), which the server uses for gate-informed aggregation.
"""
from flcore.clients.clientseg import clientSeg


class clientFedSHA(clientSeg):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.gate_mass = None               # (M,) gate mass per adapter in the last round

    def train(self):
        self.ensure_organ()
        self.model.train()
        f = self.apply_lr(self.optimizer)
        print(f"Client {self.id} | {self.modality} {self.organ} | lr x{f:.2f}")
        R_sum, R_n = None, 0

        def record_gates(out, y):
            nonlocal R_sum, R_n
            R = self.model.block_routing()
            if R is not None:
                R_sum, R_n = (R if R_sum is None else R_sum + R), R_n + 1
            return None

        self.run_epochs(record_gates)
        self.gate_mass = (R_sum / R_n).mean(0) if R_n else None          # mean over batches, then over blocks
        if R_n:
            print(f"Client {self.id} | gate probabilities per block (mean over this round):\n"
                  + self.model.routing_report(R_sum / R_n), flush=True)
