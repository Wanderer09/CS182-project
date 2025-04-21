import math

import torch
import random

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "sine_regression": SineRegression,  
        "sine2cosine":Sine2Cosine,
        "sine2periodic":Sine2Periodic,
        "poly_function_regression":PolyFunctionRegression,
        "sine":Sine,
        "hard_sine_regression":HardSineRegression,
        "hard_sine2sawtooth":HardSine2sawtooth,
        "hard_sine2square":HardSine2square,
        "hard_sine2tanh":HardSine2Tanh,
        "tanh_inverse_regression": TanhInverseRegression,
        "poly2tanhregression":Poly2TanhRegression,
        "sine2poly":Sine2PolyRegression,
        "hard_sine2poly":HardSine2PolyRegression,
        "tanh_poly_regression":TanhPolyRegression,
        "poly_to_bounded_regression":PolyToBoundedRegression,
        "poly_to_softsign_regression":PolyToSoftsignRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
'''
class SineRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(SineRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            # 每个任务一个 A 向量，尺寸是 (batch_size, n_dims, 1)
            self.A = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.A = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "A" in pool_dict
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]

    def evaluate(self, xs_b):
        A = self.A.to(xs_b.device)  # shape: (batch_size, n_dims, 1)
        xsA = (xs_b @ A)[:, :, 0]   # shape: (batch_size, num_points)
        ys_b = torch.sin(xsA) * self.scale
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "A": torch.randn(num_tasks, n_dims, 1)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class Sine2Cosine(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.A = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.A = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "A" in pool_dict
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]

    def evaluate(self, xs_b, mode="train"):
        A = self.A.to(xs_b.device)
        xsA = (xs_b @ A)[:, :, 0]
        if mode == "train":
            return torch.sin(xsA) * self.scale
        elif mode == "test":
            return torch.cos(xsA) * self.scale
        else:
            raise ValueError(f"Unknown mode {mode}")

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "A": torch.randn(num_tasks, n_dims, 1)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

    
def sample_periodic_function():
    funcs = [
        #(lambda x: torch.cos(x), "cos"),
        #(lambda x: torch.sin(x + 1.0), "sin+1"),
        #(lambda x: torch.sin(2 * x), "sin2x"),
        #(lambda x: torch.sign(torch.sin(x)), "square"),
        (lambda x: ((x % (2 * math.pi)) / (2 * math.pi)), "sawtooth"),
    ]
    return random.choice(funcs)


class Sine2Periodic(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.A = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.A = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "A" in pool_dict
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]

    def evaluate(self, xs_b, mode="train"):
        A = self.A.to(xs_b.device)
        xsA = (xs_b @ A)[:, :, 0]

        if mode == "train":
            return torch.sin(xsA) * self.scale
        elif mode == "test":
            self.g, self.g_name = sample_periodic_function()
            return self.g(xsA) * self.scale
        else:
            raise ValueError(f"Unknown mode {mode}")

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "A": torch.randn(num_tasks, n_dims, 1)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class PolyFunctionRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, degree=3, scale=1.0):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.degree = degree

        if pool_dict is None and seeds is None:
            # 多项式系数 shape: (batch_size, degree + 1, n_dims)
            self.coeffs = torch.randn(batch_size, degree + 1, n_dims)
        elif seeds is not None:
            self.coeffs = torch.zeros(batch_size, degree + 1, n_dims)
            generator = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.coeffs[i] = torch.randn(degree + 1, n_dims, generator=generator)
        else:
            assert "coeffs" in pool_dict
            indices = torch.randperm(len(pool_dict["coeffs"]))[:batch_size]
            self.coeffs = pool_dict["coeffs"][indices]

    def evaluate(self, xs_b, mode="train"):
        # 输入 shape: (batch_size, n_points, n_dims)
        powers = [xs_b ** i for i in range(self.degree + 1)]  # list of (b, p, d)
        result = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i, x_pow in enumerate(powers):
            term = (x_pow * self.coeffs[:, i].unsqueeze(1).to(xs_b.device)).sum(dim=2)
            result += term
        return result * self.scale

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, degree=3, **kwargs):
        return {
            "coeffs": torch.randn(num_tasks, degree + 1, n_dims)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
'''
def sample_periodic_function():
    funcs = [
        (lambda x: ((x % (2 * math.pi)) / (2 * math.pi)), "sawtooth"),
    ]
    return random.choice(funcs)


class SineRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(SineRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.A = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.A = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "A" in pool_dict
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]

    def evaluate(self, xs_b, mode="train"):
        A = self.A.to(xs_b.device)
        xsA = (xs_b @ A)[:, :, 0]
        return torch.sin(xsA) * self.scale

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {"A": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class Sine2Cosine(SineRegression):
    def evaluate(self, xs_b, mode="train"):
        A = self.A.to(xs_b.device)
        xsA = (xs_b @ A)[:, :, 0]
        if mode == "train":
            return torch.sin(xsA) * self.scale
        elif mode == "test":
            return torch.cos(xsA) * self.scale
        else:
            raise ValueError(f"Unknown mode {mode}")


class Sine2Periodic(SineRegression):
    def evaluate(self, xs_b, mode="train"):
        A = self.A.to(xs_b.device)
        xsA = (xs_b @ A)[:, :, 0]

        if mode == "train":
            return torch.sin(xsA) * self.scale
        elif mode == "test":
            self.g, self.g_name = sample_periodic_function()
            return self.g(xsA) * self.scale
        else:
            raise ValueError(f"Unknown mode {mode}")


class PolyFunctionRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, degree=3, scale=1.0):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.degree = degree

        if pool_dict is None and seeds is None:
            self.coeffs = torch.randn(batch_size, degree + 1, n_dims)
        elif seeds is not None:
            self.coeffs = torch.zeros(batch_size, degree + 1, n_dims)
            generator = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.coeffs[i] = torch.randn(degree + 1, n_dims, generator=generator)
        else:
            assert "coeffs" in pool_dict
            indices = torch.randperm(len(pool_dict["coeffs"]))[:batch_size]
            self.coeffs = pool_dict["coeffs"][indices]

    def evaluate(self, xs_b, mode="train"):
        powers = [xs_b ** i for i in range(self.degree + 1)]
        result = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i, x_pow in enumerate(powers):
            term = (x_pow * self.coeffs[:, i].unsqueeze(1).to(xs_b.device)).sum(dim=2)
            result += term
        return result * self.scale

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, degree=3, **kwargs):
        return {"coeffs": torch.randn(num_tasks, degree + 1, n_dims)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class Sine(Task):
    """
    正弦函数任务。
    每个任务生成一个正弦函数 y = A * sin(B * x + C) + D。
    """
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        A_range=(0.1, 5.0),
        B_range=(0.1, 5.0),
         C_range=(0, math.pi),
        D_range=(-1.0, 1.0),
     ):
        """
        初始化正弦函数任务。
        - 振幅范围 (A)
        - 频率范围 (B)
        - 相位范围 (C)
        - 偏移范围 (D)
        """
        super(Sine, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.A_range = A_range
        self.B_range = B_range
        self.C_range = C_range
        self.D_range = D_range

        if pool_dict is None and seeds is None:
            # 随机生成参数
            self.A_b = torch.empty(batch_size).uniform_(*A_range)
            self.B_b = torch.empty(batch_size).uniform_(*B_range)
            self.C_b = torch.empty(batch_size).uniform_(*C_range)
            self.D_b = torch.empty(batch_size).uniform_(*D_range)
        elif seeds is not None:
            generator = torch.Generator()
            assert len(seeds) == batch_size
            self.A_b = torch.empty(batch_size).uniform_(*A_range)
            self.B_b = torch.empty(batch_size).uniform_(*B_range)
            self.C_b = torch.empty(batch_size).uniform_(*C_range)
            self.D_b = torch.empty(batch_size).uniform_(*D_range)

            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A_b[i] = torch.empty(1).uniform_(
                    *A_range, generator=generator
                )
                self.B_b[i] = torch.empty(1).uniform_(
                    *B_range, generator=generator
                )
                self.C_b[i] = torch.empty(1).uniform_(
                    *C_range, generator=generator
                )
                self.D_b[i] = torch.empty(1).uniform_(
                    *D_range, generator=generator
                )
        else:
            raise NotImplementedError
        
    def evaluate(self, xs_b):
        """
        根据输入 xs_b 计算正弦函数的输出 ys_b。
        - xs_b: 输入张量，形状为 (batch_size, n_points, n_dims)
        返回:
        - ys_b: 输出张量，形状为 (batch_size, n_points)
        """
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            # 计算正弦函数 y = A * sin(B * x + C) + D。
            ys_b[i] = (
                self.A_b[i] * torch.sin(
                    self.B_b[i] * xs_b[i, :, 0] + self.C_b[i]
                ) + self.D_b[i]
            )
        return ys_b
        
    @staticmethod
    def get_metric():
        return squared_error
        
    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
import math
import torch
from tasks import Task  # 假设你已有 Task 基类

class HardSineRegression(Task):
    """
    更复杂的正弦回归任务：y = A * sin(Bx + C) + D
    支持 mode="train"/"test"，保持一致性。
    """
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        A_range=(0.5, 2.0),
        B_range=(0.5, 2.0),
        C_range=(0.0, math.pi),
        D_range=(-1.0, 1.0),
    ):
        super(HardSineRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.A_range = A_range
        self.B_range = B_range
        self.C_range = C_range
        self.D_range = D_range

        if pool_dict is None and seeds is None:
            self.A = torch.empty(batch_size).uniform_(*A_range)
            self.B = torch.empty(batch_size).uniform_(*B_range)
            self.C = torch.empty(batch_size).uniform_(*C_range)
            self.D = torch.empty(batch_size).uniform_(*D_range)
        elif seeds is not None:
            generator = torch.Generator()
            self.A = torch.zeros(batch_size)
            self.B = torch.zeros(batch_size)
            self.C = torch.zeros(batch_size)
            self.D = torch.zeros(batch_size)
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.empty(1).uniform_(*A_range, generator=generator)
                self.B[i] = torch.empty(1).uniform_(*B_range, generator=generator)
                self.C[i] = torch.empty(1).uniform_(*C_range, generator=generator)
                self.D[i] = torch.empty(1).uniform_(*D_range, generator=generator)
        else:
            assert all(k in pool_dict for k in ["A", "B", "C", "D"])
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]
            self.B = pool_dict["B"][indices]
            self.C = pool_dict["C"][indices]
            self.D = pool_dict["D"][indices]

    def evaluate(self, xs_b, mode="train"):
        """
        xs_b: shape (batch_size, n_points, n_dims)
        返回: shape (batch_size, n_points)
        """
        xs_proj = xs_b.mean(dim=2)  # 将多维特征投影成标量输入
        A, B, C, D = self.A.to(xs_b.device), self.B.to(xs_b.device), self.C.to(xs_b.device), self.D.to(xs_b.device)
        ys_b = A[:, None] * torch.sin(B[:, None] * xs_proj + C[:, None]) + D[:, None]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "A": torch.empty(num_tasks).uniform_(0.5, 2.0),
            "B": torch.empty(num_tasks).uniform_(0.5, 2.0),
            "C": torch.empty(num_tasks).uniform_(0.0, math.pi),
            "D": torch.empty(num_tasks).uniform_(-1.0, 1.0),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


def generate_periodic_function(wave_type="sawtooth", T=2*math.pi, low=0.0, high=1.0):
    """
    返回一个周期函数 f(x)，满足：
        - f(x + T) = f(x)
        - 值域在 [low, high]
        - 波形由 wave_type 决定
    """
    def normalize(val):
        return low + (high - low) * val

    if wave_type == "sawtooth":
        def f(x):
            phase = (x % T) / T  # Normalize to [0, 1]
            return normalize(phase)

    elif wave_type == "square":
        def f(x):
            phase = (x % T) / T
            return normalize((phase < 0.5).float())

    elif wave_type == "triangle":
        def f(x):
            phase = (x % T) / T
            triangle = 2 * torch.abs(phase - 0.5)  # V shape
            return normalize(1 - triangle)

    elif wave_type == "sin":
        def f(x):
            return normalize(0.5 * (torch.sin(2 * math.pi * x / T) + 1))

    elif wave_type == "cos":
        def f(x):
            return normalize(0.5 * (torch.cos(2 * math.pi * x / T) + 1))

    else:
        raise ValueError(f"Unknown wave_type: {wave_type}")

    return f

class HardSine2sawtooth(HardSineRegression):
    def __init__(self, *args, periodic_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.periodic_kwargs = periodic_kwargs or {
            "wave_type": "sawtooth", "T": 2*math.pi, "low": 0.0, "high": 1.0
        }

    def evaluate(self, xs_b, mode="train"):
        xs_proj = xs_b.mean(dim=2)
        A, B, C, D = self.A.to(xs_b.device), self.B.to(xs_b.device), self.C.to(xs_b.device), self.D.to(xs_b.device)

        if mode == "train":
            return A[:, None] * torch.sin(B[:, None] * xs_proj + C[:, None]) + D[:, None]
        elif mode == "test":
            g = generate_periodic_function(**self.periodic_kwargs)
            x_input = B[:, None] * xs_proj + C[:, None]
            return A[:, None] * g(x_input) + D[:, None]
        else:
            raise ValueError(f"Unknown mode {mode}")

class HardSine2square(HardSineRegression):
    def __init__(self, *args, periodic_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.periodic_kwargs = periodic_kwargs or {
            "wave_type": "square", "T": 2*math.pi, "low": 0.0, "high": 1.0
        }

    def evaluate(self, xs_b, mode="train"):
        xs_proj = xs_b.mean(dim=2)
        A, B, C, D = self.A.to(xs_b.device), self.B.to(xs_b.device), self.C.to(xs_b.device), self.D.to(xs_b.device)

        if mode == "train":
            return A[:, None] * torch.sin(B[:, None] * xs_proj + C[:, None]) + D[:, None]
        elif mode == "test":
            g = generate_periodic_function(**self.periodic_kwargs)
            x_input = B[:, None] * xs_proj + C[:, None]
            return A[:, None] * g(x_input) + D[:, None]
        else:
            raise ValueError(f"Unknown mode {mode}")

class HardSine2Tanh(HardSineRegression):
    """
    训练阶段使用 y = A * sin(Bx + C) + D
    测试阶段使用 y = A * tanh(Bx + C) + D
    """
    def evaluate(self, xs_b, mode="train"):
        xs_proj = xs_b.mean(dim=2)
        A, B, C, D = self.A.to(xs_b.device), self.B.to(xs_b.device), self.C.to(xs_b.device), self.D.to(xs_b.device)
        x_input = B[:, None] * xs_proj + C[:, None]

        if mode == "train":
            return A[:, None] * torch.sin(x_input) + D[:, None]
        elif mode == "test":
            return A[:, None] * torch.tanh(x_input) + D[:, None]
        else:
            raise ValueError(f"Unknown mode {mode}")

class TanhInverseRegression(Task):
    """
    Tanh 反函数回归任务（y = A * arctanh(Bx + C) + D）
    注意：arctanh 的定义域为 (-1, 1)，所以输入需进行约束。
    """
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        A_range=(0.5, 2.0),
        B_range=(0.1, 0.9),  # 防止 Bx + C 超出 (-1, 1)
        C_range=(-0.5, 0.5),
        D_range=(-1.0, 1.0),
    ):
        super(TanhInverseRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.A_range = A_range
        self.B_range = B_range
        self.C_range = C_range
        self.D_range = D_range

        if pool_dict is None and seeds is None:
            self.A = torch.empty(batch_size).uniform_(*A_range)
            self.B = torch.empty(batch_size).uniform_(*B_range)
            self.C = torch.empty(batch_size).uniform_(*C_range)
            self.D = torch.empty(batch_size).uniform_(*D_range)
        elif seeds is not None:
            generator = torch.Generator()
            self.A = torch.zeros(batch_size)
            self.B = torch.zeros(batch_size)
            self.C = torch.zeros(batch_size)
            self.D = torch.zeros(batch_size)
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.empty(1).uniform_(*A_range, generator=generator)
                self.B[i] = torch.empty(1).uniform_(*B_range, generator=generator)
                self.C[i] = torch.empty(1).uniform_(*C_range, generator=generator)
                self.D[i] = torch.empty(1).uniform_(*D_range, generator=generator)
        else:
            assert all(k in pool_dict for k in ["A", "B", "C", "D"])
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]
            self.B = pool_dict["B"][indices]
            self.C = pool_dict["C"][indices]
            self.D = pool_dict["D"][indices]

    def evaluate(self, xs_b, mode="train"):
        xs_proj = xs_b.mean(dim=2)  # shape: (b, p)
        A, B, C, D = self.A.to(xs_b.device), self.B.to(xs_b.device), self.C.to(xs_b.device), self.D.to(xs_b.device)
        x_input = B[:, None] * xs_proj + C[:, None]

        # clip 输入以避免数值不稳定（arctanh 只定义在 (-1, 1)）
        x_input_clipped = torch.clamp(x_input, min=-0.999, max=0.999)
        ys_b = A[:, None] * 0.5 * torch.log((1 + x_input_clipped) / (1 - x_input_clipped)) + D[:, None]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "A": torch.empty(num_tasks).uniform_(0.5, 2.0),
            "B": torch.empty(num_tasks).uniform_(0.1, 0.9),
            "C": torch.empty(num_tasks).uniform_(-0.5, 0.5),
            "D": torch.empty(num_tasks).uniform_(-1.0, 1.0),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class Poly2TanhRegression(Task):
    """
    在多项式函数上训练，在 tanh(多项式函数) 上测试。
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, degree=3, scale=1.0):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.degree = degree

        if pool_dict is None and seeds is None:
            self.coeffs = torch.randn(batch_size, degree + 1, n_dims)
        elif seeds is not None:
            self.coeffs = torch.zeros(batch_size, degree + 1, n_dims)
            generator = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.coeffs[i] = torch.randn(degree + 1, n_dims, generator=generator)
        else:
            assert "coeffs" in pool_dict
            indices = torch.randperm(len(pool_dict["coeffs"]))[:batch_size]
            self.coeffs = pool_dict["coeffs"][indices]

    def evaluate(self, xs_b, mode="train"):
        powers = [xs_b ** i for i in range(self.degree + 1)]  # list of (b, p, d)
        result = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i, x_pow in enumerate(powers):
            term = (x_pow * self.coeffs[:, i].unsqueeze(1).to(xs_b.device)).sum(dim=2)
            result += term
        poly_output = result * self.scale

        if mode == "train":
            return poly_output
        elif mode == "test":
            return torch.tanh(poly_output)
        else:
            raise ValueError(f"Unknown mode {mode}")

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, degree=3, **kwargs):
        return {
            "coeffs": torch.randn(num_tasks, degree + 1, n_dims)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

import torch
import math
from tasks import Task, squared_error, mean_squared_error

class Sine2PolyRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, degree=3, scale=1.0):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.degree = degree

        # A 用于 sine 训练函数
        if pool_dict is None and seeds is None:
            self.A = torch.randn(self.b_size, self.n_dims, 1)
            self.coeffs = torch.randn(self.b_size, degree + 1, n_dims)
        elif seeds is not None:
            self.A = torch.zeros(self.b_size, self.n_dims, 1)
            self.coeffs = torch.zeros(self.b_size, degree + 1, n_dims)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.randn(self.n_dims, 1, generator=generator)
                self.coeffs[i] = torch.randn(degree + 1, n_dims, generator=generator)
        else:
            assert "A" in pool_dict and "coeffs" in pool_dict
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]
            self.coeffs = pool_dict["coeffs"][indices]

    def evaluate(self, xs_b, mode="train"):
        if mode == "train":
            A = self.A.to(xs_b.device)
            xsA = (xs_b @ A)[:, :, 0]
            return torch.sin(xsA) * self.scale
        elif mode == "test":
            powers = [xs_b ** i for i in range(self.degree + 1)]  # [(b, p, d)]
            result = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
            for i, x_pow in enumerate(powers):
                term = (x_pow * self.coeffs[:, i].unsqueeze(1).to(xs_b.device)).sum(dim=2)
                result += term
            return result * self.scale
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, degree=3, **kwargs):
        return {
            "A": torch.randn(num_tasks, n_dims, 1),
            "coeffs": torch.randn(num_tasks, degree + 1, n_dims)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
import torch
import math
from tasks import Task, squared_error, mean_squared_error

class HardSine2PolyRegression(Task):
    """
    训练：HardSine（y = A * sin(Bx + C) + D）
    测试：Poly（x）
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None,
                 A_range=(0.5, 2.0), B_range=(0.5, 2.0), C_range=(0.0, math.pi),
                 D_range=(-1.0, 1.0), degree=3, scale=1.0):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.A_range = A_range
        self.B_range = B_range
        self.C_range = C_range
        self.D_range = D_range
        self.degree = degree
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.A = torch.empty(batch_size).uniform_(*A_range)
            self.B = torch.empty(batch_size).uniform_(*B_range)
            self.C = torch.empty(batch_size).uniform_(*C_range)
            self.D = torch.empty(batch_size).uniform_(*D_range)
            self.coeffs = torch.randn(batch_size, degree + 1, n_dims)
        elif seeds is not None:
            generator = torch.Generator()
            self.A = torch.zeros(batch_size)
            self.B = torch.zeros(batch_size)
            self.C = torch.zeros(batch_size)
            self.D = torch.zeros(batch_size)
            self.coeffs = torch.zeros(batch_size, degree + 1, n_dims)
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.empty(1).uniform_(*A_range, generator=generator)
                self.B[i] = torch.empty(1).uniform_(*B_range, generator=generator)
                self.C[i] = torch.empty(1).uniform_(*C_range, generator=generator)
                self.D[i] = torch.empty(1).uniform_(*D_range, generator=generator)
                self.coeffs[i] = torch.randn(degree + 1, n_dims, generator=generator)
        else:
            assert all(k in pool_dict for k in ["A", "B", "C", "D", "coeffs"])
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]
            self.B = pool_dict["B"][indices]
            self.C = pool_dict["C"][indices]
            self.D = pool_dict["D"][indices]
            self.coeffs = pool_dict["coeffs"][indices]

    def evaluate(self, xs_b, mode="train"):
        xs_proj = xs_b.mean(dim=2)  # shape: (b, p)
        A, B, C, D = self.A.to(xs_b.device), self.B.to(xs_b.device), self.C.to(xs_b.device), self.D.to(xs_b.device)

        if mode == "train":
            return A[:, None] * torch.sin(B[:, None] * xs_proj + C[:, None]) + D[:, None]
        elif mode == "test":
            # 计算 poly(x)
            powers = [xs_b ** i for i in range(self.degree + 1)]  # [(b, p, d)]
            result = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
            for i, x_pow in enumerate(powers):
                term = (x_pow * self.coeffs[:, i].unsqueeze(1).to(xs_b.device)).sum(dim=2)
                result += term
            return result * self.scale
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, degree=3, **kwargs):
        return {
            "A": torch.empty(num_tasks).uniform_(0.5, 2.0),
            "B": torch.empty(num_tasks).uniform_(0.5, 2.0),
            "C": torch.empty(num_tasks).uniform_(0.0, math.pi),
            "D": torch.empty(num_tasks).uniform_(-1.0, 1.0),
            "coeffs": torch.randn(num_tasks, degree + 1, n_dims),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class TanhPolyRegression(Task):
    """
    训练 + 测试：y = tanh(poly(x))
    poly(x) 是多项式：c₀ + c₁x + c₂x² + ... + cₙxⁿ
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, degree=3, scale=1.0):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.degree = degree

        if pool_dict is None and seeds is None:
            self.coeffs = torch.randn(batch_size, degree + 1, n_dims)
        elif seeds is not None:
            self.coeffs = torch.zeros(batch_size, degree + 1, n_dims)
            generator = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.coeffs[i] = torch.randn(degree + 1, n_dims, generator=generator)
        else:
            assert "coeffs" in pool_dict
            indices = torch.randperm(len(pool_dict["coeffs"]))[:batch_size]
            self.coeffs = pool_dict["coeffs"][indices]

    def evaluate(self, xs_b, mode="train"):
        powers = [xs_b ** i for i in range(self.degree + 1)]  # list of shape (b, p, d)
        result = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i, x_pow in enumerate(powers):
            term = (x_pow * self.coeffs[:, i].unsqueeze(1).to(xs_b.device)).sum(dim=2)
            result += term
        output = torch.tanh(result * self.scale)
        return output

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, degree=3, **kwargs):
        return {
            "coeffs": torch.randn(num_tasks, degree + 1, n_dims)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class PolyToBoundedRegression(Task):
    """
    训练和验证都在 f(poly(x)) 上进行，其中 f(x) = x / sqrt(1 + x^2)
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, degree=3, scale=1.0):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.degree = degree
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.coeffs = torch.randn(batch_size, degree + 1, n_dims)
        elif seeds is not None:
            self.coeffs = torch.zeros(batch_size, degree + 1, n_dims)
            generator = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.coeffs[i] = torch.randn(degree + 1, n_dims, generator=generator)
        else:
            assert "coeffs" in pool_dict
            indices = torch.randperm(len(pool_dict["coeffs"]))[:batch_size]
            self.coeffs = pool_dict["coeffs"][indices]

    def evaluate(self, xs_b, mode="train"):
        powers = [xs_b ** i for i in range(self.degree + 1)]  # list of (b, p, d)
        result = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i, x_pow in enumerate(powers):
            term = (x_pow * self.coeffs[:, i].unsqueeze(1).to(xs_b.device)).sum(dim=2)
            result += term
        poly_output = result * self.scale

        # 映射到有界函数 f(x) = x / sqrt(1 + x^2)
        bounded_output = poly_output / torch.sqrt(1 + poly_output ** 2)
        return bounded_output

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, degree=3, **kwargs):
        return {
            "coeffs": torch.randn(num_tasks, degree + 1, n_dims)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

def smooth_clip_softsign(x, alpha=5.0):
    return alpha * x / (1 + torch.abs(x))

class PolyToSoftsignRegression(Task):
    """
    在 f(poly(x)) 上训练与测试，f(x) = alpha * x / (1 + |x|)，输出更均匀、便于训练
    """
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, degree=3, scale=1.0, alpha=5.0):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.degree = degree
        self.alpha = alpha

        if pool_dict is None and seeds is None:
            self.coeffs = torch.randn(batch_size, degree + 1, n_dims)
        elif seeds is not None:
            self.coeffs = torch.zeros(batch_size, degree + 1, n_dims)
            generator = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.coeffs[i] = torch.randn(degree + 1, n_dims, generator=generator)
        else:
            assert "coeffs" in pool_dict
            indices = torch.randperm(len(pool_dict["coeffs"]))[:batch_size]
            self.coeffs = pool_dict["coeffs"][indices]

    def evaluate(self, xs_b, mode="train"):
        powers = [xs_b ** i for i in range(self.degree + 1)]  # (b, p, d)
        result = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i, x_pow in enumerate(powers):
            term = (x_pow * self.coeffs[:, i].unsqueeze(1).to(xs_b.device)).sum(dim=2)
            result += term
        poly_output = result * self.scale
        return self.alpha * poly_output / (1 + torch.abs(poly_output))

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, degree=3, **kwargs):
        return {
            "coeffs": torch.randn(num_tasks, degree + 1, n_dims)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
