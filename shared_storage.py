class SharedStorage:
    def __init__(self):
        self.network = None

    def latest_checkpoint(self):
        return self.network

    def save_network(self, net):
        self.network = net
