"""
FedHCA2 client (competing method; Lu et al., CVPR 2024) on the common base network (frozen MedSAM + dense MSA
adapters + MedSAM decoder). As in the official code, local training is the plain task loss (segmentation:
focal + Dice), with no extra term; all of FedHCA2 happens at the server (serverHCA2.py: hyper conflict-averse
aggregation of the encoder adapters, hyper cross-attention aggregation of the decoder trunk, learnable hyper-weights).
Losses, optimiser, learning rates and evaluation are those of every other method (clientSeg via clientFedAvg).
"""
from flcore.clients.clientFedAvg import clientFedAvg


class clientHCA2(clientFedAvg):
    def train(self):
        self.model.train()
        f = self.apply_lr(self.optimizer)
        print(f"Client {self.id} | {self.modality} {self.organ} | lr x{f:.2f} | FedHCA2")
        self.run_epochs()
