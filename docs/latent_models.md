# 🔍 Latent Variable Models: LCA and LPA

Latent variable models aim to uncover hidden structures in data by modeling unobserved (latent) classes or profiles. Two common approaches are:

## 📘 Latent Class Analysis (LCA)

LCA is a **categorical latent variable model** used to identify subgroups within a population based on discrete observed variables.

### 🔢 Mathematical Model

Let \( X = (X_1, X_2, ..., X_p) \) be a set of observed categorical variables, and \( Z \) be a latent class variable with \( K \) classes.

The probability of an observation is:



\[
P(X) = \sum_{k=1}^{K} P(Z = k) \prod_{j=1}^{p} P(X_j \mid Z = k)
\]



Where:
- \( P(Z = k) \) is the class prior (mixing proportion)
- \( P(X_j \mid Z = k) \) is the conditional probability of each variable given the class

## 📗 Latent Profile Analysis (LPA)

LPA is the **continuous analogue** of LCA. It models latent profiles based on continuous observed variables.

### 🔢 Mathematical Model

Assuming \( X \in \mathbb{R}^p \), the model is:



\[
P(X) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(X \mid \mu_k, \Sigma_k)
\]



Where:
- \( \pi_k \) is the mixing proportion for class \( k \)
- \( \mu_k \), \( \Sigma_k \) are the mean and covariance of class \( k \)

LPA assumes that each latent profile follows a multivariate normal distribution.

---

## 🧠 Use Cases

- **LCA**: Used when input variables are categorical (e.g., symptoms: yes/no).
- **LPA**: Used when input variables are continuous (e.g., lab values, age).

Both models are useful for identifying **phenotypes**, **behavioral patterns**, or **risk profiles** in clinical and social science data.
