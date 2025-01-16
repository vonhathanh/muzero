class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.children = {}
        self.reward = 0
        self.value_sum = 0
        self.hidden_state = None

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count