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
