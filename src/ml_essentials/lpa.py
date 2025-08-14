from stepmix.stepmix import StepMix
import numpy as np

# Simulated continuous data: 3 features
X = np.random.normal(size=(500, 3))

# Fit LPA model with 3 latent profiles
model = StepMix(n_components=3, measurement='continuous', n_steps=1)
model.fit(X)

# Predict profile membership
profiles = model.predict(X)
print("Profile assignments:", profiles[:10])
