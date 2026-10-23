"""Local baseline client: the base network trained on the client's own data only (see clientFedAvg)."""
from flcore.clients.clientFedAvg import clientFedAvg


class clientLocal(clientFedAvg):
    pass
