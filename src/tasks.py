import math

import torch


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
        "sine": Sine,
        "polynomial_regression": PolynomialRegression,
        "sine2exp": Sine2Exp,
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
    
class Sine(Task):
    """
    Sine task
    Aorted. Train please use Sine2Exp.
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
        super(Sine, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.A_range = A_range
        self.B_range = B_range
        self.C_range = C_range
        self.D_range = D_range

        if pool_dict is None and seeds is None:
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
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
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

class PolynomialRegression(Task):
    def __init__(
        self,
        n_dims,                     
        batch_size,                 
        degree=None,                
        coeff_range=(-1.0, 1.0),    
        noise_std=0.0,              
        renormalize_ys=False,       
        pool_dict=None,             
        seeds=None,
    ):
        assert n_dims == 1, "Polynomial task only supports 1D input x."
        super(PolynomialRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        
        if degree is None:
            degree = 3

        self.degree = degree
        self.coeff_range = coeff_range
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

        if pool_dict is None and seeds is None:
            self.coeffs = torch.empty(batch_size, degree + 1).uniform_(*coeff_range)
        
        elif seeds is not None:
            self.coeffs = torch.zeros(batch_size, degree + 1)
            generator = torch.Generator()
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.coeffs[i] = torch.empty(degree + 1).uniform_(*coeff_range, generator=generator)
        
        else:
            assert "coeffs" in pool_dict
            indices = torch.randperm(len(pool_dict["coeffs"]))[:batch_size]
            self.coeffs = pool_dict["coeffs"][indices]

    def evaluate(self, xs_b):
        x = xs_b.squeeze(-1)  
        B, N = x.shape

        powers = torch.stack([x ** k for k in range(self.degree + 1)], dim=-1) 

        coeffs = self.coeffs.to(x.device).unsqueeze(1)

        ys = torch.sum(powers * coeffs, dim=-1)

        if self.noise_std > 0:
            ys += torch.randn_like(ys) * self.noise_std

        if self.renormalize_ys:
            ys = ys * math.sqrt(self.degree + 1) / ys.std()

        return ys

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, degree=3, coeff_range=(-1.0, 1.0), **kwargs):
        return {
            "coeffs": torch.empty(num_tasks, degree + 1).uniform_(*coeff_range)
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
def generate_periodic_function(wave_type="sawtooth", T=2*math.pi, low=-1.0, high=1.0):
    """
    Generate a periodic function based on the specified wave type.
    Args:
        wave_type (str): Type of wave function to generate. Options: "sawtooth", "square", "triangle", "sin", "cos".
        T (float): Period of the wave function.
        low (float): Minimum value of the wave function.
        high (float): Maximum value of the wave function.
    Returns:
        function: A function that takes an input x and returns the corresponding wave value.
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

def piecewise_high_freq_sine_gauss(x):
    """
    Piecewise high-frequency sine function.
    """
    out = torch.zeros_like(x)
    mask1 = x < -1.5
    mask2 = (x >= -1.5) & (x < -0.5)
    mask3 = (x >= -0.5) & (x < 0.5)
    mask4 = (x >= 0.5) & (x < 1.5)
    mask5 = x >= 1.5

    out[mask1] = torch.sin(2 * torch.pi * x[mask1])
    out[mask2] = torch.sin(8 * torch.pi * x[mask2])
    out[mask3] = torch.sin(16 * torch.pi * x[mask3])
    out[mask4] = torch.sin(8 * torch.pi * x[mask4])
    out[mask5] = torch.sin(2 * torch.pi * x[mask5])

    return out

    
class Sine2Exp(Task):
    """
    Train: Sine
    Eval: Sine, Exp, Periodic, Decision Tree, Piecewise Sine, Combo Nonlinear
    """

    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        A_range=(0.1, 5.0),
        B_range=(0.1, 5.0),
        C_range=(0.0, math.pi),
        D_range=(-1.0, 1.0),
        E_range=(1.0, 1.0),
        F_range=(1.0, 1.0),
        G_range=(0.0, 0.0),
        periodic_config = None,
        tree_depth=4,
    ):
        super().__init__(n_dims, batch_size, pool_dict, seeds)
        
        self.periodic_config = {
            "wave_type": "square",
            "T": 2 * math.pi,
            "low": -1.0,
            "high": 1.0,
        }
        self.tree_depth = tree_depth

        if pool_dict is None and seeds is None:
            self.A = torch.empty(batch_size).uniform_(*A_range)
            self.B = torch.empty(batch_size).uniform_(*B_range)
            self.C = torch.empty(batch_size).uniform_(*C_range)
            self.D = torch.empty(batch_size).uniform_(*D_range)
            self.E = torch.empty(batch_size).uniform_(*E_range)
            self.F = torch.empty(batch_size).uniform_(*F_range)
            self.G = torch.empty(batch_size).uniform_(*G_range)

        elif seeds is not None:
            generator = torch.Generator()
            self.A = torch.zeros(batch_size)
            self.B = torch.zeros(batch_size)
            self.C = torch.zeros(batch_size)
            self.D = torch.zeros(batch_size)
            self.E = torch.zeros(batch_size)
            self.F = torch.zeros(batch_size)
            self.G = torch.zeros(batch_size)
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.A[i] = torch.empty(1).uniform_(*A_range, generator=generator)
                self.B[i] = torch.empty(1).uniform_(*B_range, generator=generator)
                self.C[i] = torch.empty(1).uniform_(*C_range, generator=generator)
                self.D[i] = torch.empty(1).uniform_(*D_range, generator=generator)
                self.E[i] = torch.empty(1).uniform_(*E_range, generator=generator)
                self.F[i] = torch.empty(1).uniform_(*F_range, generator=generator)
                self.G[i] = torch.empty(1).uniform_(*G_range, generator=generator)

        elif pool_dict is not None:
            assert all(k in pool_dict for k in ["A", "B", "C", "D", "E", "F", "G"])
            indices = torch.randperm(len(pool_dict["A"]))[:batch_size]
            self.A = pool_dict["A"][indices]
            self.B = pool_dict["B"][indices]
            self.C = pool_dict["C"][indices]
            self.D = pool_dict["D"][indices]
            self.E = pool_dict["E"][indices]
            self.F = pool_dict["F"][indices]
            self.G = pool_dict["G"][indices]

        else:
            raise ValueError("Must specify either pool_dict or seeds (or neither).")
        
        # generate decision tree
        self.dt_tensor = torch.randint(
            low=0, high=n_dims, size=(batch_size, 2 ** (tree_depth + 1) - 1)
        )
        self.target_tensor = torch.randn(batch_size, 2 ** (tree_depth + 1) - 1)
        self.dt_thresholds = torch.empty(batch_size, 2 ** (tree_depth + 1) - 1).uniform_(-1.5, 1.5)

    def evaluate(self, xs_b, mode="period"):        # PLEASE CHANGE THIS MODE WHEN DOING TRAINING AND EVAL #
        xs_proj = xs_b.mean(dim=2)  # shape: (b, p)
        if mode == "sine":
            return self.A[:, None].to(xs_b.device) * torch.sin(
                self.B[:, None].to(xs_b.device) * xs_proj + self.C[:, None].to(xs_b.device)
            ) + self.D[:, None].to(xs_b.device)
        elif mode == "exp":
            return self.E[:, None].to(xs_b.device) * torch.exp(
                self.F[:, None].to(xs_b.device) * xs_proj
            ) + self.G[:, None].to(xs_b.device)
        elif mode == "period":
            g = generate_periodic_function(**self.periodic_config)
            x_input = self.B[:, None] * xs_proj + self.C[:, None]
            return g(x_input)
        elif mode == "decision_tree":
            xs_proj = xs_b.mean(dim=2)          # (B, T)
            B, T = xs_proj.shape
            ys_b = torch.zeros(B, T, device=xs_b.device)

            dt_tensor = self.dt_tensor.to(xs_b.device)
            dt_thresholds = self.dt_thresholds.to(xs_b.device)
            target_tensor = self.target_tensor.to(xs_b.device)

            for i in range(B):
                dt = dt_tensor[i]
                thresholds = dt_thresholds[i]
                target = target_tensor[i]
                cur_nodes = torch.zeros(T, dtype=torch.long, device=xs_b.device)

                for d in range(self.tree_depth):
                    feature_id = dt[cur_nodes]
                    thres = thresholds[cur_nodes] 

                    x_values = xs_b[i, torch.arange(T), feature_id]  # shape: (T,)
                    decisions = (x_values > thres).long()
                    cur_nodes = 2 * cur_nodes + 1 + decisions

                ys_b[i] = target[cur_nodes]

            return ys_b
        elif mode == "piecewise_sine":
            x_proj = xs_b.mean(dim=2)
            return piecewise_high_freq_sine_gauss(x_proj)
        elif mode == "combo_nonlinear":
            x_proj = xs_b.mean(dim=2)  # shape: (B, T)
            
            offset = 2.0
            scale = 1.5
            x_shifted = scale * (x_proj + offset)
            
            return torch.sin(2 * x_shifted ** 2)

        else:
            raise ValueError(f"Unknown mode {mode}")

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):
        return {
            "A": torch.empty(num_tasks).uniform_(0.5, 2.0),
            "B": torch.empty(num_tasks).uniform_(0.5, 2.0),
            "C": torch.empty(num_tasks).uniform_(0.0, math.pi),
            "D": torch.empty(num_tasks).uniform_(-1.0, 1.0),
            "E": torch.empty(num_tasks).uniform_(0.5, 2.0),
            "F": torch.empty(num_tasks).uniform_(0.1, 0.9),
            "G": torch.empty(num_tasks).uniform_(-1.0, 1.0),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error