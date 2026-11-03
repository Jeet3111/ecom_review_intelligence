"""
nn.py  -  From-scratch neural-network toolkit (NumPy only, no TensorFlow / PyTorch).

Contains
    * activation functions + derivatives
    * cost (loss) functions: Binary Cross-Entropy, Mean Squared Error, Categorical Cross-Entropy
    * Perceptron (classic Rosenblatt learning rule)
    * MLP  (forward propagation + back-propagation + mini-batch training loop)
    * Optimizer (variants of gradient descent):
          gd, momentum, nesterov, adagrad, rmsprop, adam
      (Batch GD / SGD / Mini-batch GD are obtained through `batch_size` in MLP.fit)
    * numerical gradient checking (proves back-propagation is correct)
"""
import json
import time
import numpy as np

EPS = 1e-7  # clipping constant for logs (safe for float32)


# --------------------------------------------------------------------------- #
# 1. Activation functions
# --------------------------------------------------------------------------- #
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# name -> (f(z), f'(z) written in terms of (z, a=f(z)))
ACTIVATIONS = {
    "relu": (lambda z: np.maximum(0.0, z),
             lambda z, a: (z > 0).astype(z.dtype)),
    "leaky_relu": (lambda z: np.where(z > 0, z, 0.01 * z),
                   lambda z, a: np.where(z > 0, 1.0, 0.01).astype(z.dtype)),
    "sigmoid": (sigmoid, lambda z, a: a * (1.0 - a)),
    "tanh": (np.tanh, lambda z, a: 1.0 - a * a),
    "linear": (lambda z: z, lambda z, a: np.ones_like(z)),
}


# --------------------------------------------------------------------------- #
# 2. Cost functions
# --------------------------------------------------------------------------- #
def loss_value(name, Y, A):
    """Average cost over the batch.  Y = targets, A = network output."""
    if name == "bce":      # J = -1/m * sum[ y log(a) + (1-y) log(1-a) ]
        A = np.clip(A, EPS, 1 - EPS)
        return float(-np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A)))
    if name == "mse":      # J = 1/m * sum (a - y)^2
        return float(np.mean((A - Y) ** 2))
    if name == "cce":      # J = -1/m * sum_i sum_k y_ik log(a_ik)
        return float(-np.mean(np.sum(Y * np.log(np.clip(A, EPS, 1.0)), axis=1)))
    raise ValueError(f"unknown loss {name}")


def output_delta(loss, out_act, Y, A):
    """dJ/dz for the output layer (the 'error signal' that starts back-prop)."""
    m = Y.shape[0]
    if loss == "bce" and out_act == "sigmoid":      # (a - y)/m   <- sigmoid cancels
        return (A - Y) / m
    if loss == "cce" and out_act == "softmax":      # (a - y)/m
        return (A - Y) / m
    if loss == "mse" and out_act == "sigmoid":      # 2(a-y)a(1-a)/m  <- saturates!
        return 2.0 * (A - Y) * A * (1.0 - A) / m
    if loss == "mse" and out_act == "linear":
        return 2.0 * (A - Y) / m
    raise ValueError(f"unsupported combination loss={loss}, out_act={out_act}")


# --------------------------------------------------------------------------- #
# 3. Variants of gradient descent (update rules)
# --------------------------------------------------------------------------- #
class Optimizer:
    """
    gd        : p <- p - lr*g                                  (plain gradient descent)
    momentum  : v <- mu*v - lr*g ;           p <- p + v
    nesterov  : v <- mu*v - lr*g ;           p <- p - mu*v_prev + (1+mu)*v
    adagrad   : G <- G + g^2 ;               p <- p - lr*g/(sqrt(G)+eps)
    rmsprop   : s <- rho*s + (1-rho)*g^2 ;   p <- p - lr*g/(sqrt(s)+eps)
    adam      : m,v moving averages (+ bias correction) ; p <- p - lr*m_hat/(sqrt(v_hat)+eps)
    Whether this is *Batch*, *Stochastic* or *Mini-batch* GD is decided by how many
    samples are used to compute g (see MLP.fit(batch_size=...)).
    """
    NAMES = ("gd", "momentum", "nesterov", "adagrad", "rmsprop", "adam")

    def __init__(self, name="adam", lr=1e-3, mu=0.9, beta1=0.9, beta2=0.999, rho=0.9, eps=1e-8):
        if name not in self.NAMES:
            raise ValueError(f"optimizer must be one of {self.NAMES}")
        self.name, self.lr, self.mu = name, lr, mu
        self.b1, self.b2, self.rho, self.eps = beta1, beta2, rho, eps
        self.t = 0
        self.s1, self.s2 = {}, {}

    def step(self, params, grads):
        self.t += 1
        n, lr = self.name, self.lr
        for i, (p, g) in enumerate(zip(params, grads)):
            if n == "gd":
                p -= lr * g
            elif n == "momentum":
                v = self.s1.setdefault(i, np.zeros_like(p))
                v *= self.mu
                v -= lr * g
                p += v
            elif n == "nesterov":
                v = self.s1.setdefault(i, np.zeros_like(p))
                v_prev = v.copy()
                v *= self.mu
                v -= lr * g
                p += -self.mu * v_prev + (1 + self.mu) * v
            elif n == "adagrad":
                G = self.s1.setdefault(i, np.zeros_like(p))
                G += g * g
                p -= lr * g / (np.sqrt(G) + self.eps)
            elif n == "rmsprop":
                s = self.s1.setdefault(i, np.zeros_like(p))
                s *= self.rho
                s += (1 - self.rho) * g * g
                p -= lr * g / (np.sqrt(s) + self.eps)
            elif n == "adam":
                m = self.s1.setdefault(i, np.zeros_like(p))
                v = self.s2.setdefault(i, np.zeros_like(p))
                m *= self.b1
                m += (1 - self.b1) * g
                v *= self.b2
                v += (1 - self.b2) * g * g
                m_hat = m / (1 - self.b1 ** self.t)
                v_hat = v / (1 - self.b2 ** self.t)
                p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


# --------------------------------------------------------------------------- #
# 4. Perceptron (single neuron, step activation)
# --------------------------------------------------------------------------- #
class Perceptron:
    """Rosenblatt perceptron:  y_hat = step(w.x + b);  if wrong:  w += lr*y*x ,  b += lr*y  (y in {-1,+1})."""

    def __init__(self, lr=1.0, epochs=20, seed=0):
        self.lr, self.epochs, self.seed = lr, epochs, seed
        self.w, self.b = None, 0.0
        self.errors_per_epoch = []

    def fit(self, X, y01):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = 2 * np.asarray(y01).reshape(-1) - 1
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_per_epoch = []
        for _ in range(self.epochs):
            errors = 0
            for i in rng.permutation(len(X)):
                if y[i] * (X[i] @ self.w + self.b) <= 0:      # misclassified -> update
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]
                    errors += 1
            self.errors_per_epoch.append(errors)
            if errors == 0:                                      # converged (linearly separable)
                break
        return self

    def decision(self, X):
        return np.asarray(X, dtype=np.float64) @ self.w + self.b

    def predict(self, X):
        return (self.decision(X) > 0).astype(int)


# --------------------------------------------------------------------------- #
# 5. Multi-layer perceptron with back-propagation
# --------------------------------------------------------------------------- #
class MLP:
    def __init__(self, sizes, hidden_act="relu", out_act="sigmoid", loss="bce",
                 l2=0.0, seed=0, dtype=np.float32):
        self.sizes = list(sizes)
        self.hidden_act, self.out_act, self.loss, self.l2 = hidden_act, out_act, loss, l2
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            act = hidden_act if i < len(sizes) - 2 else out_act
            if act in ("relu", "leaky_relu"):                       # He initialisation
                W = rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            else:                                                    # Xavier / Glorot
                lim = np.sqrt(6.0 / (fan_in + fan_out))
                W = rng.uniform(-lim, lim, (fan_in, fan_out))
            self.W.append(W.astype(dtype))
            self.b.append(np.zeros((1, fan_out), dtype=dtype))

    # ----- forward propagation ------------------------------------------------
    def forward(self, X, store=False):
        A = X
        Zs, As = [], [X]
        L = len(self.W)
        for i in range(L):
            Z = A @ self.W[i] + self.b[i]
            if i < L - 1:
                A = ACTIVATIONS[self.hidden_act][0](Z)
            elif self.out_act == "softmax":
                A = softmax(Z)
            else:
                A = ACTIVATIONS[self.out_act][0](Z)
            if store:
                Zs.append(Z)
                As.append(A)
        if store:
            self._Z, self._A = Zs, As
        return A

    # ----- back-propagation ---------------------------------------------------
    def backward(self, Y):
        """Chain rule from the output layer to the first layer. Needs forward(store=True) first."""
        L = len(self.W)
        dZ = output_delta(self.loss, self.out_act, Y, self._A[-1])
        gW, gb = [None] * L, [None] * L
        for i in reversed(range(L)):
            gW[i] = self._A[i].T @ dZ + self.l2 * self.W[i]
            gb[i] = dZ.sum(axis=0, keepdims=True)
            if i > 0:
                dA = dZ @ self.W[i].T                                         # propagate error backwards
                dZ = dA * ACTIVATIONS[self.hidden_act][1](self._Z[i - 1], self._A[i])
        return gW, gb

    # ----- helpers --------------------------------------------------------------
    def cost(self, X, Y):
        J = loss_value(self.loss, Y, self.forward(X))
        if self.l2:
            J += 0.5 * self.l2 * sum(float((W ** 2).sum()) for W in self.W)
        return J

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        P = self.forward(X)
        if P.shape[1] == 1:
            return (P[:, 0] >= 0.5).astype(int)
        return P.argmax(axis=1)

    def _acc(self, X, Y):
        P = self.forward(X)
        if P.shape[1] == 1:
            return float(np.mean((P >= 0.5).astype(int) == Y))
        return float(np.mean(P.argmax(1) == Y.argmax(1)))

    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # ----- training loop --------------------------------------------------------
    def fit(self, X, Y, X_val=None, Y_val=None, epochs=20, batch_size=64, optimizer="adam",
            lr=1e-3, seed=0, patience=None, track_grad=False, verbose=False, **opt_kw):
        """
        batch_size = None / >= n  -> Batch Gradient Descent
        batch_size = 1            -> Stochastic Gradient Descent
        otherwise                 -> Mini-batch Gradient Descent
        """
        n = X.shape[0]
        bs = n if (not batch_size or batch_size >= n) else int(batch_size)
        rng = np.random.default_rng(seed)
        opt = Optimizer(optimizer, lr, **opt_kw)
        params = self.W + self.b
        hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                "grad_norm": [], "epoch_time": [], "updates": 0, "diverged": False,
                "best_epoch": None, "stopped_epoch": None}
        best, best_state, wait = np.inf, None, 0
        for ep in range(epochs):
            t0 = time.perf_counter()
            idx = rng.permutation(n) if bs < n else np.arange(n)
            for s in range(0, n, bs):
                ids = idx[s:s + bs]
                self.forward(X[ids], store=True)
                gW, gb = self.backward(Y[ids])
                opt.step(params, gW + gb)
                hist["updates"] += 1
            tl = self.cost(X, Y)
            if not np.isfinite(tl):
                hist["diverged"] = True
                hist["train_loss"].append(float("nan"))
                break
            hist["train_loss"].append(tl)
            hist["train_acc"].append(self._acc(X, Y))
            if track_grad:
                self.forward(X, store=True)
                gW, gb = self.backward(Y)
                hist["grad_norm"].append(float(np.sqrt(sum((g ** 2).sum() for g in gW + gb))))
            if X_val is not None:
                vl = self.cost(X_val, Y_val)
                hist["val_loss"].append(vl)
                hist["val_acc"].append(self._acc(X_val, Y_val))
                if patience:
                    if vl < best - 1e-5:
                        best, wait = vl, 0
                        best_state = [p.copy() for p in params]
                        hist["best_epoch"] = ep + 1
                    else:
                        wait += 1
                        if wait >= patience:
                            hist["stopped_epoch"] = ep + 1
                            hist["epoch_time"].append(time.perf_counter() - t0)
                            break
            hist["epoch_time"].append(time.perf_counter() - t0)
            if verbose:
                print(f"  epoch {ep + 1:3d}  loss={tl:.4f}  acc={hist['train_acc'][-1]:.4f}"
                      + (f"  val_loss={hist['val_loss'][-1]:.4f}" if X_val is not None else ""))
        if best_state is not None:                      # restore the best weights (early stopping)
            for p, bp in zip(params, best_state):
                p[...] = bp
        return hist

    # ----- persistence ----------------------------------------------------------
    def save(self, path):
        meta = dict(sizes=self.sizes, hidden_act=self.hidden_act, out_act=self.out_act, loss=self.loss)
        arrays = {f"W{i}": W for i, W in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        m = cls(meta["sizes"], meta["hidden_act"], meta["out_act"], meta["loss"])
        m.W = [z[f"W{i}"] for i in range(len(meta["sizes"]) - 1)]
        m.b = [z[f"b{i}"] for i in range(len(meta["sizes"]) - 1)]
        return m


# --------------------------------------------------------------------------- #
# 6. Gradient checking  (analytic back-prop gradient vs. numerical gradient)
# --------------------------------------------------------------------------- #
def gradient_check(hidden_act="tanh", loss="bce", out_act="sigmoid", n_out=1, seed=0, eps=1e-5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(12, 5))
    if n_out == 1:
        Y = rng.integers(0, 2, size=(12, 1)).astype(float)
    else:
        Y = np.eye(n_out)[rng.integers(0, n_out, 12)]
    net = MLP([5, 7, 4, n_out], hidden_act, out_act, loss, seed=seed, dtype=np.float64)
    net.forward(X, store=True)
    gW, gb = net.backward(Y)
    analytic = np.concatenate([g.ravel() for g in gW + gb])
    numeric = []
    for p in net.W + net.b:
        flat = p.reshape(-1)
        for k in range(flat.size):
            old = flat[k]
            flat[k] = old + eps
            jp = net.cost(X, Y)
            flat[k] = old - eps
            jm = net.cost(X, Y)
            flat[k] = old
            numeric.append((jp - jm) / (2 * eps))
    numeric = np.array(numeric)
    rel = np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12)
    return float(rel)
