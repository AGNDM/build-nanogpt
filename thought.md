# Learning Rate

- Cosine -> linear?
- No warm up?
- Evaluate the outcome
- Why min_lr is not 0? Underflow especially on the smaller mantissa float on GPUs
  - Ideally, learning rate should be 0 in the end. Is it possible to have the GPUs switch to another float number representation some time before the underflow?

# Dataset

- 2 epochs on first 5B dataset vs 1 epoch on 10B dataset
  - Which one is better?
- Is randomization really helping us to improve the performance?

# Architecture

- Group the decoder blocks and the weight of each multihead attention in the same group is the same
  - Is it faster?
- Bigger/Smaller?

# Model Size

- nanogpt_bigger

# Initialization

- What if we simply initialize all weights to be 1?
