# 🧬 Phenotypic Clustering — Dengue Data Analysis

This repository presents a **technical and educational reference** for phenotypic clustering analysis based on Dengue case data from the Brazilian SINAN database. It is part of the NOIS organization's data science initiatives.

The goal is to identify distinct phenotypes of Dengue cases using Latent Class Analysis technique, enabling better understanding of disease patterns and supporting public health decision-making.

This code repository stores the draft codes that later will comprise the [ISARIC analytical pipeline](https://github.com/noispuc/isaric3.0-rapid)

## 🧠 What you'll find here:

Example case of StepMix package use for latent class analysis using SINAN's Dengue data.
Adjusted codes for grid search, and bootstrapping for latent class analysis optimization.
— the grid search
    —  store objects,
    —  reports graphically the major metrics - AIC, BIC, SABIC, CAIC, entropy and relative entropy -,
    -  alongside the information on convergence status for each number of classes k tested.
- the bootstrapping method also was improved to
    -  store the metrics - AIC, BIC, SABIC, CAIC, entropy and relative entropy
    -  bootstrapped selection rate for metrics AIC, BIC,
    -  the percentage of convergence,
    -  and display them in a visual way
    -  so users can have all the information together to enable users to make a guided decision on which number of classes k to use
- visualization of the results
    - relevant results are captured from StepMix object and displayed for the user to guide their interpretations
    - radar plots are used to visualize the adjusted conditional probabilities of each class on the features used to adjust them
- example use of the codes (grid search)

## why to adjust new codes beyond StepMix?

Latent class analysis, while not a completely new concept, has early foundations dating back to the 1950s and 1960s. It has become more feasible in recent decades due to advancements in computational efficiency and the availability of open-source packages. Nevertheless, currently, the options for performing latent class analysis in Python are limited, with [StepMix](https://github.com/Labo-Lacourse/stepmix) among the primary tools available, and it still falls short compared to the wide range of packages available in R. 

One of the major limitations, for non-programming-savvy users, was the lack of relevant information to support more guided decision-making during the optimisation and results interpretation stages. The implementation of Stepmix 2.2.3 focused heavily on displaying log-likelihood results. Nevertheless, it was crucial to make hidden details such as AIC and BIC available to users, empowering them to utilize the comprehensive information for informed decision-making. Although intermediate and advanced Python users could understand and use these ideas, as well as get them from python/stepmix objects, the ISARIC analytical platform is made for a different group of users, mainly epidemiological researchers. Therefore, we dedicated efforts to enhance usability for these users.

This usability enhancement, specifically for the case of LCA (Latent Class Analysis), aims to provide epidemiological researchers with:

- A solid foundation for more informed decision-making regarding the number of classes to utilize. This includes:
    - Graphical representations of metrics to clearly illustrate how these metrics change as the number of classes is increased or decreased.
    - Information on the convergence status for each setting (considering factors such as random seed and maximum iterations).
    - Inclusion of bootstrapped confidence intervals, in addition to the p-values from log-likelihood tests.
    - Access to visual and tabulated reports on the results of sequential runs.
    
- Graphical reports on class descriptives.

## Example code

1. [Using the `optimize_stepMix()` class to conduct class enumeration for Latent Class Analysis](jupyterFiles/1_gridSearch.ipynb)
2. [exploring the grid search results from `optimize_stepMix()` class to actually define on class number.](jupyterFiles/1b_Inspect_gridSearchResults.ipynb)
3. [Conducting bootstraped likelihood ratio test and getting other bootstrapped methods for class enumeration](jupyterFiles/2_Bootstrap5.ipynb)
4. [Continuing bootstrap process when it crashes because issues like energy shortage](jupyterFiles/2b_Bootstrap5_restoreAndContinue.ipynb) 


## 🛠️ Technologies Used

This repository integrates several key technologies to support data analysis:
- Python – Programming language powering analytics and automation
- [Stepmix](https://github.com/Labo-Lacourse/stepmix) – Python library for latent class and mixture modeling
- [Scikit-learn](https://scikit-learn.org/stable/) - Python library for machine learning
- [scipy](https://scipy.org/) - Fundamental algorithms for scientific computing in python
- Matplotlib
- Jupyter Notebooks – Interactive development environments for executing and understand statistics techniques
- MkDocs - Generate documentation

## ⚙️ Requirements
- Python 3.10+
- Virtual environment recommended

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
