import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CustomRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Minimal custom Random Forest classifier for benchmarking."""

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _fit_one_tree(self, X, y, seed):
        rng = np.random.RandomState(seed)
        if self.bootstrap:
            indices = rng.choice(X.shape[0], size=X.shape[0], replace=True)
        else:
            indices = np.arange(X.shape[0])

        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=seed,
        )
        tree.fit(X[indices], y[indices])
        return tree

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")

        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_estimators)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one_tree)(X, y, int(seed)) for seed in seeds
        )

        self.feature_importances_ = np.mean(
            np.vstack([tree.feature_importances_ for tree in self.estimators_]), axis=0
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, accept_sparse=False)
        probs = np.mean(
            np.stack([tree.predict_proba(X) for tree in self.estimators_], axis=0),
            axis=0,
        )
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
