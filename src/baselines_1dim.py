import torch

class ZeroEstimator:
    def __init__(self):
        self.name = "Zero Estimator"

    def __call__(self, xs, ys):
        B, T, D = xs.shape
        K = ys.shape[1]
        N = T - K
        return torch.zeros((B, N), device=xs.device)


class MeanEstimator:
    def __init__(self):
        self.name = "Mean Estimator"

    def __call__(self, xs, ys):
        B, T, D = xs.shape
        K = ys.shape[1]
        N = T - K
        if K == 0:
            return torch.zeros((B, N), device=xs.device)
        mean_y = ys.mean(dim=1, keepdim=True)  # (B, 1)
        return mean_y.expand(-1, N)            # (B, N)


class LinearEstimator:
    def __init__(self):
        self.name = "Linear Regression"

    def __call__(self, xs, ys):
        B, T, D = xs.shape
        K = ys.shape[1]
        N = T - K
        if K == 0:
            return torch.zeros((B, N), device=xs.device)

        x_context = xs[:, :K, 0]  # (B, K)
        x_query = xs[:, K:, 0]    # (B, N)

        preds = []
        for i in range(B):
            x_i = x_context[i]
            y_i = ys[i]
            A = torch.stack([x_i, torch.ones_like(x_i)], dim=1)
            coeff, *_ = torch.linalg.lstsq(A, y_i.unsqueeze(-1))
            w, b = coeff[0], coeff[1]
            y_pred = w * x_query[i] + b
            preds.append(y_pred)

        return torch.stack(preds)


class OraclePolynomialEstimator:
    def __init__(self, degree=3):
        self.degree = degree
        self.name = f"Oracle Poly Fit (deg={degree})"

    def __call__(self, xs, ys):
        B, T, D = xs.shape
        K = ys.shape[1]
        N = T - K
        if K == 0:
            return torch.zeros((B, N), device=xs.device)

        x_context = xs[:, :K, 0]  # (B, K)
        x_query = xs[:, K:, 0]    # (B, N)

        preds = []
        for i in range(B):
            x_i = x_context[i]
            y_i = ys[i]
            A = torch.stack([x_i ** j for j in range(self.degree + 1)], dim=1)
            coeff, *_ = torch.linalg.lstsq(A, y_i.unsqueeze(-1))
            query_feats = torch.stack([x_query[i] ** j for j in range(self.degree + 1)], dim=1)
            y_pred = (query_feats @ coeff).squeeze(-1)
            preds.append(y_pred)

        return torch.stack(preds)
