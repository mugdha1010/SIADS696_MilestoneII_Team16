"""Model registry. Add a model by writing one small factory function and registering it."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# name -> (factory(cfg, pos_weight) -> estimator, preprocessor kind)
REGISTRY = {}


def register(name, kind):
    def deco(fn):
        REGISTRY[name] = (fn, kind)
        return fn
    return deco


@register("dummy", "sparse")
def _dummy(cfg, pw):
    return DummyClassifier(strategy="prior")


@register("logreg", "sparse")
def _logreg(cfg, pw):
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, solver="lbfgs")


@register("linear_svm", "sparse")
def _svm(cfg, pw):
    return CalibratedClassifierCV(LinearSVC(C=0.1, class_weight="balanced"), cv=3, method="sigmoid")


@register("random_forest", "dense")
def _rf(cfg, pw):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=cfg.n_jobs,
                                  random_state=cfg.random_state)


@register("hist_gb", "dense")
def _hgb(cfg, pw):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=63,
                                          class_weight="balanced", early_stopping=True,
                                          random_state=cfg.random_state)


@register("xgboost", "sparse")
def _xgb(cfg, pw):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.5, min_child_weight=5, scale_pos_weight=pw,
                         tree_method="hist", eval_metric="aucpr", n_jobs=cfg.n_jobs,
                         random_state=cfg.random_state)


@register("lightgbm", "sparse")
def _lgbm(cfg, pw):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.5, scale_pos_weight=pw,
                          n_jobs=cfg.n_jobs, random_state=cfg.random_state, verbose=-1)


@register("knn", "dense")
def _knn(cfg, pw):
    return KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=cfg.n_jobs)


def get_model(name, cfg, pos_weight):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    factory, kind = REGISTRY[name]
    return factory(cfg, pos_weight), kind
