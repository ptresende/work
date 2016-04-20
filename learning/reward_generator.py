class RewardGenerator():
    def __init__(self):
        self._penalty = -1
        self._reward = 1

    def evaluate(self, label, action):
        if label is None:
            if action is not None:
                return self._penalty
            else:
                return 0
        if label != action:
            return self._penalty
        else:
            return self._reward
