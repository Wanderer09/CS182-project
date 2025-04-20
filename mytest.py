import torch

x = torch.tensor([
    0.4197, -0.0802, 1.3803, -0.5237, 0.0642, -0.3115, 1.4164, 0.4071,
    -0.4294, 0.5481, 1.5761, 0.3618, 1.0405, -1.4926, 0.0462, -0.9608,
    0.0076, -0.4007, -1.9117, -1.4444
])

# Step 1: clamp to [-0.999, 0.999]
x_clipped = torch.clamp(x, -0.999, 0.999)

# Step 2: apply arctanh
y_values = torch.arctanh(x_clipped)

# Step 3: take mean
y_mean = y_values.mean()
print(y_mean)
