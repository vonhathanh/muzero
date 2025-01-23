
class ActionHistory(object):
    """Simple history container used inside the search.

    Only used to keep track of the actions executed.
    """

    def __init__(self, history: list, action_space: list[int]):
        self.history = list(history)
        self.action_space = action_space

    def clone(self):
        return ActionHistory(self.history, self.action_space)

    def add_action(self, action: int):
        self.history.append(action)

    def last_action(self) -> int:
        return self.history[-1]
