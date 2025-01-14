import random

from args import Args


class ReplayBuffer:
    def __init__(self, args: Args):
        self.size = args.replay_buffer_size
        self.batch_size = args.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = self.sample_games()
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_games(self):
        # Sample game from buffer either uniformly or according to some priority.
        return random.choices(self.buffer, k=self.batch_size)

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        pass
