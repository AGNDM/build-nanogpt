# Learning Rate

- Cosine -> linear? _Code done_ _Script done_
- No warm up? _Code done_ _Script done_
- Evaluate the outcome
- Why min_lr is not 0? Underflow especially on the smaller mantissa float on GPUs
  - Ideally, learning rate should be 0 in the end. Is it possible to have the GPUs switch to another float number representation some time before the underflow?

# Dataset

- 2 epochs on first 5B dataset vs 1 epoch on 10B dataset _TODO_
  - Which one is better?
- Is randomization really helping us to improve the performance? _Code done_ _Script done_ _Job submitted_

# Architecture

- Group the decoder blocks and the weight of each multihead attention in the same group is the same _TODO_
  - Is it faster?
- Bigger _Code done_ _Script done_ _Job testing_
- Smaller _Code done_ _Script done_

# Initialization

- What if we simply initialize all weights to be 1? _TODO_
