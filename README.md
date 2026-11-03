# Indian E-commerce Review Intelligence
**Spam / fake-review detection + sentiment + topics, using a neural network built from scratch (NumPy)**
M.Tech AI-ML · LCA1 — *Design and Deployment of an AI-Based Web Application for Classification using Neural Networks*

```
Input message  →  AI model (MLP)  →  Spam / Not Spam   (+ sentiment, topics, dashboard)
```

Everything required by the assignment brief **plus** the three Deep-Learning topics is inside this zip:

| Requirement (brief / DLN topic) | Where it is |
|---|---|
| 1. Dataset preparation, pre-processing, feature extraction | `reviewintel/synth.py`, `reviewintel/features.py` |
| 2. **Perceptron** (+ limitations, XOR) | `reviewintel/nn.py → Perceptron`, figure `04_xor_perceptron_vs_mlp.png` |
| 3. Neural network, forward propagation, activations (ReLU, Sigmoid, Tanh, Leaky-ReLU, Softmax) | `reviewintel/nn.py → MLP.forward`, `ACTIVATIONS` |
| 4. **Cost functions** in NN (BCE, MSE, Categorical CE) + loss-vs-epoch plots | `reviewintel/nn.py → loss_value / output_delta`, figures `02_…`, `06_…`, `11_…` |
| 5. **Back-propagation** + convergence + numerical gradient check | `reviewintel/nn.py → MLP.backward`, `gradient_check` |
| 6. **Variants of gradient descent** (Batch, SGD, Mini-batch, Momentum, Nesterov, AdaGrad, RMSProp, Adam) | `reviewintel/nn.py → Optimizer`, figures `07_…`, `08_…` |
| 7. Hyper-parameter tuning (6 hyper-parameters) | `run_pipeline.py` step 8, figure `09_…`, `outputs/tables/hyperparameter_results.csv` |
| 8. Accuracy, precision, recall, F1, confusion matrix, ROC | `outputs/results.json`, figures `12_…`, `13_…`, `14_…` |
| 9. Web application (message → Spam / Not Spam) + dashboard | `webapp/` |
| 10. Google Cloud deployment | `Dockerfile`, `deploy_gcp.sh`, section 9 below |
| 11. Report | `report/Review_Intelligence_Report.docx` |

> **About the data.** The build environment has no internet, so the project ships a **synthetic** but realistic generator of Indian e-commerce reviews (English + Hinglish, 6 categories, 120 products, ratings, prices, dates, six spam/fake types, hard negatives, 1.5 % label noise). Results therefore describe *this* dataset, not real Flipkart/Amazon traffic. To use real data, see "Using a real dataset" below — the whole pipeline, dashboard and report regenerate automatically.

---

## 1. Requirements
* Python 3.9 – 3.13, ~1 GB RAM, no GPU. Works on Windows, macOS, Linux.
* Packages: `numpy pandas scikit-learn matplotlib flask gunicorn python-docx` (see `requirements.txt`).
  The **web app itself needs only** `flask numpy gunicorn` (`requirements-app.txt`).

## 2. Step-by-step: run everything

### Fastest: one command
```bash
# Linux / macOS / Git-Bash
./run_all.sh              # add --quick for a 1-minute smoke test
# Windows
run_all.bat
```
This creates a virtual-env, installs packages, trains + runs every experiment, runs the self-tests, builds the Word report and starts the web app at **http://127.0.0.1:8080**.

### Or manually, step by step
```bash
# 0. unzip and enter the folder
cd ecom_review_intelligence

# 1. (recommended) virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. install packages
pip install -r requirements.txt

# 3. generate data, train, run all experiments, save models + figures   (≈ 8–12 min; --quick ≈ 1 min)
python run_pipeline.py

# 4. self-tests (gradient check, optimisers, perceptron/XOR, web endpoints)
python -m unittest discover -s tests -v

# 5. build the Word report from the fresh results
python report/build_report.py        # -> report/Review_Intelligence_Report.docx

# 6. start the web app
python webapp/app.py                 # open http://127.0.0.1:8080
```

What step 3 prints: dataset stats → gradient-check error (~1e-10) → perceptron → baseline MLP → BCE vs MSE → 8 gradient-descent variants → hyper-parameter grids → final test metrics → sentiment head. All figures land in `outputs/figures/`, numbers in `outputs/results.json` and `outputs/tables/*.csv`, trained weights in `models/`.

## 3. Using the web app
* **Analyzer** (`/`) — paste a review, click *Analyze*. Shows Spam / Not Spam, P(spam) with the decision threshold, sentiment (genuine reviews only), topics, rule-based spam signals and the words that influenced the decision (occlusion).
* **Dashboard** (`/dashboard`) — KPI cards (products, categories, reviews, spam %, avg rating), review trend, sentiment pie, rating distribution, top positive/negative topics, price-vs-rating scatter, spam by category.
* **Model & Experiments** (`/model`) — metrics, confusion matrix, ROC, cost-function / gradient-descent / hyper-parameter figures.
* **API** — `POST /api/predict` with `{"text": "..."}` →
```bash
curl -X POST http://127.0.0.1:8080/api/predict -H "Content-Type: application/json" \
     -d '{"text":"Free recharge! click www.deal-hub.in or WhatsApp 9876501234"}'
```

## 4. Using a real dataset
Create a CSV with at least `review_text` and `is_spam` (0/1 or spam/ham). Optional columns enable more of the dashboard/sentiment: `rating, category, product_id, product_name, price_inr, review_date, platform`.
```bash
python run_pipeline.py --data my_reviews.csv
python report/build_report.py
```
Good public sources: the UCI *SMS Spam Collection*, Kaggle *Flipkart product reviews*, *Amazon India reviews*, *Yelp/Opspam fake reviews* (rename columns as above). Without a `rating` column the sentiment head is skipped automatically.

## 5. What the neural network does (short)
```
x (1510) = [ TF-IDF uni+bi-grams (1500) | 10 style features ]
   → Dense(h, ReLU) × L  → Dense(1, Sigmoid)  → P(spam)
cost  : binary cross-entropy      update : mini-batch Adam      stop : early stopping on validation cost
```
Back-propagation is hand-written and verified against numerical gradients (relative error ≈ 1e-10).

## 6. Hyper-parameters explored
learning rate · hidden neurons · hidden layers · batch size · activation function · epochs (+ decision threshold, L2).

## 7. Cloud deployment (Google Cloud Run)
1. Redeem the student coupon (link in the brief) and create/choose a project with billing enabled.
2. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and run `gcloud auth login`.
3. Make sure training was done (`models/spam_model.npz` exists), then:
```bash
PROJECT_ID=<your-project-id> ./deploy_gcp.sh          # region defaults to asia-south1 (Mumbai)
```
   or the manual equivalent:
```bash
gcloud config set project <your-project-id>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy review-intelligence --source . --region asia-south1 --allow-unauthenticated --memory 512Mi
```
4. The command prints the public **Service URL** — open it, test a spam and a genuine review, take screenshots for the report.
5. To avoid charges afterwards: `gcloud run services delete review-intelligence --region asia-south1`.

Test the container locally first (needs Docker): `docker build -t ri . && docker run -p 8080:8080 ri`.

## 8. Project structure
```
ecom_review_intelligence/
├── run_pipeline.py            # ALL experiments (data → features → perceptron → MLP → cost → GD variants → tuning → eval)
├── run_all.sh / run_all.bat   # one-command run
├── reviewintel/
│   ├── nn.py                  # ★ Perceptron, MLP, back-prop, cost functions, optimisers, gradient check
│   ├── features.py            # cleaning, TF-IDF (from scratch), style features
│   ├── synth.py               # synthetic Indian e-commerce review generator
│   ├── topics.py  dashboard.py  plots.py  inference.py
├── webapp/                    # Flask app (app.py, templates/, static/)
├── models/                    # trained weights (npz) + feature extractor (json) + meta
├── outputs/                   # results.json, figures/, tables/
├── data/reviews.csv           # generated dataset
├── report/                    # build_report.py → Review_Intelligence_Report.docx
├── tests/test_core.py
├── Dockerfile  deploy_gcp.sh  requirements*.txt
```

## 9. Troubleshooting
| Problem | Fix |
|---|---|
| `Model not found` banner in the app | run `python run_pipeline.py` first |
| Port 8080 busy | `PORT=5000 python webapp/app.py` (Windows: `set PORT=5000`) |
| Pipeline too slow | `python run_pipeline.py --quick` |
| `gcloud` permission / billing error | check the coupon is applied to the project; enable the three APIs listed above |
| Different numbers from the report | the pipeline is seeded (42) but BLAS / library versions can shift the 3rd decimal; rebuild the report after every run |
