# muzero implementation plan

- Implement muzero for toy problems first: lunar lander, cart pole
- Define & implement our own environment for muzero

# What we need to implement in muzero
- RB: Replay buffer (prioritized RB is optional)
- MCTS:
  - Selection phase
  - Expansion phase
  - Backup phase
  - Node
- Model architecture:
  - Representation network: h(x)
  - Dynamic network: g(x)
  - Policy (value) network: f(x)
  - All of them are pure neural network/transformer architecture, mo computer vision model like CNN or Resnes is involved
- Single process -> multiprocess training & data sharing
- Value transformation
- Hyperparameters selection
- Data generation phase
  - Input: raw state representation -> run MCTS -> store trajectory in RB
- Training phase
  - Random select trajectory from RB, align model's output with result from MCTS by a loss function
  - Evaluation metrics, save/loading checkpoints

# TODOs
- Understand the UCB formula, better rewrite it from scratch
- Understand how MCTS works, in detail