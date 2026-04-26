# Spaceship Titanic — Predicting Transported Passengers

My first end-to-end machine learning project, built after finishing the ML section of *Hands-On Machine Learning* (Géron).
The goal: predict which passengers were transported to another dimension based on the [Kaggle Spaceship Titanic dataset]

## Approach

### 1. Exploratory analysis
First pass through the data with `.describe()`, `.info()`, and basic plots. Three things stood out:

- **`Cabin`** was a single string in the format `Deck/Number/Side` — needed to be split into three separate features so the model could use them.
- **Expense columns** (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) were heavily right-skewed.
-  Most passengers spent little; a few spent a lot, and big spenders tended to concentrate their spending in a single service.
- **`PassengerId`** encoded group membership — passengers traveling together share the same group prefix. Family/group context likely matters for the target.

### 2. Feature engineering
- **Split `Cabin`** into `Cabin_Deck`, `Cabin_Num`, `Cabin_Side`.
- **Total expenses**: summed all spending columns into one feature, since the per-service breakdown was noisy and concentration patterns suggested the *total* is what matters.
- **`alone` flag**: derived from the `PassengerId` group prefix — 1 if the passenger has no group companions, 0 otherwise.
- **Dropped `Name`** — group/family signal is already captured by the `alone` feature, and the name strings themselves added noise.

### 3. Preprocessing pipeline
Built with `ColumnTransformer` + `Pipeline` to keep everything leakage-free across cross-validation folds:

- **Numeric features**: mean imputation → log transform (for the skewed expense feature) → standardization.
- Standardization was important because tree boosters are robust to scale, but I tested linear models too, and consistent scaling kept the comparison fair.
- **Categorical features**: most-frequent imputation → one-hot encoding with `handle_unknown="ignore"`.

### 4. Model selection
I compared three model families with the same preprocessing pipeline:

| Model | Notes |
| Logistic Regression | Fast baseline, underfit the non-linear interactions. |
| SVM (RBF kernel) | Better than LR but slow to train and tune. |
| **HistGradientBoostingClassifier** | Best accuracy, fast, handles missing values natively. **Chosen.** |

I also experimented with **KMeans clustering** as an additional feature and **PCA** for dimensionality reduction during preprocessing — neither moved the score meaningfully, so I dropped them to keep the pipeline simple.

### 5. Hyperparameter tuning
Used `GridSearchCV` to tune the boosting model. The final configuration favors a slow, well-regularized learner:

```python
HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,      # slow learner
    max_depth=5,             # shallow trees
    min_samples_leaf=20,     # avoid memorizing rare patterns
    l2_regularization=1.5,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
)


## What I learned
- **Pipelines matter.** Wrapping preprocessing inside `Pipeline` + `ColumnTransformer` is the only clean way to do cross-validation without leakage. My first attempt fit the scaler on the full training set before splitting — the score looked great and was wrong.
- **Simpler is often better.** PCA and clustering felt sophisticated but added nothing. Knowing when to remove a step is as important as knowing how to add one.
- **The model is not the bottleneck.** Most of the accuracy gain came from feature engineering (cabin split, total expenses, group flag), not from switching algorithms.

## What I'd do next
- Stack the boosting model with a Logistic Regression meta-learner.
- Investigate the ~635 false negatives in the confusion matrix — are they concentrated on a specific deck or age group?
- Re-add the per-service expense features alongside the total, and let regularization sort out which ones help.

---

*Built as part of my Applied Informatics studies at the University of Duisburg-Essen, while transitioning toward a Data Science / AI career.*
