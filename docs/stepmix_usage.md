# 🧪 Using Stepmix for Latent Class and Profile Analysis

[Stepmix](https://github.com/stepmix/stepmix) is a Python library for fitting latent variable models using the Expectation-Maximization (EM) algorithm.

## 📦 Installation

```bash
pip install stepmix

📊 Example: Latent Class Analysis (LCA)
import numpy as np
from stepmix.stepmix import StepMix

# Simulated categorical data: 3 binary variables
X = np.random.randint(0, 2, size=(500, 3))

# Fit LCA model with 2 latent classes
model = StepMix(n_components=2, measurement='categorical', n_steps=1)
model.fit(X)

# Predict class membership
classes = model.predict(X)
probs = model.predict_proba(X)

print("Class assignments:", classes[:10])
print("Class probabilities:", probs[:10])

📈 Example: Latent Profile Analysis (LPA)
from stepmix.stepmix import StepMix

# Simulated continuous data: 3 features
X = np.random.normal(size=(500, 3))

# Fit LPA model with 3 latent profiles
model = StepMix(n_components=3, measurement='continuous', n_steps=1)
model.fit(X)

# Predict profile membership
profiles = model.predict(X)
print("Profile assignments:", profiles[:10])

⚙️ Parameters
n_components: Number of latent classes/profiles

measurement: 'categorical' for LCA, 'continuous' for LPA

n_steps: Number of EM steps (usually 1 for standard models)

📌 Notes
Stepmix supports mixed models (categorical + continuous).

You can extract parameters like class probabilities, means, and covariances for interpretation.
