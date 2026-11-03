"""Quick self-tests.  Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reviewintel.nn import MLP, Optimizer, Perceptron, gradient_check   # noqa: E402
from reviewintel.features import FeatureExtractor, clean_text           # noqa: E402


class TestNN(unittest.TestCase):
    def test_backprop_matches_numerical_gradient(self):
        for h, l, o, k in [("tanh", "bce", "sigmoid", 1), ("relu", "bce", "sigmoid", 1), ("sigmoid", "mse", "sigmoid", 1),
                           ("relu", "cce", "softmax", 3)]:
            self.assertLess(gradient_check(h, l, o, k), 1e-6, f"{h}/{l}/{o}")

    def test_perceptron_learns_linearly_separable(self):
        rng = np.random.default_rng(0)
        X = np.r_[rng.normal(2, .5, (50, 2)), rng.normal(-2, .5, (50, 2))]
        y = np.r_[np.ones(50), np.zeros(50)]
        self.assertEqual((Perceptron(epochs=50).fit(X, y).predict(X) == y).mean(), 1.0)

    def test_perceptron_cannot_solve_xor_but_mlp_can(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
        y = np.array([0, 1, 1, 0])
        self.assertLess((Perceptron(lr=.1, epochs=200).fit(X, y).predict(X) == y).mean(), 1.0)
        m = MLP([2, 6, 1], "tanh", seed=3, dtype=np.float64)
        m.fit(X, y.reshape(-1, 1).astype(float), epochs=3000, batch_size=None, lr=.02)
        self.assertTrue((m.predict(X) == y).all())

    def test_every_optimizer_decreases_the_cost(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 5)).astype(np.float32)
        y = (X[:, :1] + X[:, 1:2] > 0).astype(np.float32)
        lrs = dict(gd=.5, momentum=.1, nesterov=.1, adagrad=.1, rmsprop=.01, adam=.01)
        for name in Optimizer.NAMES:
            m = MLP([5, 8, 1], seed=0)
            h = m.fit(X, y, epochs=15, batch_size=32, optimizer=name, lr=lrs[name])
            self.assertLess(h["train_loss"][-1], h["train_loss"][0], name)


class TestFeatures(unittest.TestCase):
    def test_clean_text_keeps_spam_tokens(self):
        t = clean_text("WIN NOW!!! visit www.dealz.in or call 9876543210 @promo")
        self.assertIn("urltoken", t)
        self.assertIn("phonetoken", t)
        self.assertIn("handletoken", t)

    def test_shapes(self):
        texts = ["good product", "bad product", "good delivery", "bad delivery"] * 5
        fx = FeatureExtractor(max_features=10, min_df=2).fit(texts)
        self.assertEqual(fx.transform(texts).shape, (20, fx.n_features))


@unittest.skipUnless((ROOT / "models" / "spam_model.npz").exists(), "run run_pipeline.py first")
class TestApp(unittest.TestCase):
    def setUp(self):
        from webapp.app import app
        self.c = app.test_client()

    def test_pages(self):
        for url in ("/", "/dashboard", "/model", "/health"):
            self.assertEqual(self.c.get(url).status_code, 200, url)

    def test_predict_spam_and_genuine(self):
        s = self.c.post("/api/predict", json={"text": "Free recharge of Rs 500, click www.abc-deals.in or WhatsApp 9876501234"}).get_json()
        g = self.c.post("/api/predict", json={"text": "Delivery was quick and the battery lasts a full day. Happy with it."}).get_json()
        self.assertTrue(s["is_spam"])
        self.assertFalse(g["is_spam"])
        self.assertIn(g["sentiment"]["label"], ("Positive", "Neutral", "Negative"))

    def test_empty_input(self):
        self.assertEqual(self.c.post("/api/predict", json={"text": "  "}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
