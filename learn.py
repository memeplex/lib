import logging
import re
import tempfile

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.utils import _safe_indexing as idx

from .base import Bundle, as_many, rand, replace, try_import

lgb = try_import("lightgbm")
xgb = try_import("xgboost")
optuna = try_import("optuna")

logger = logging.getLogger(__name__)

default_cv = KFold(5, shuffle=True)
default_storage = tempfile.mkstemp(prefix=f"{__name__}.search-")[1]


def config_sklearn(pandas_output=True):
    if pandas_output:
        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html
        set_config(transform_output="pandas")


def cv(
    est,
    X,
    y,
    w=None,
    cv=default_cv,
    scoring="r2",
    score_train=False,
    trial=None,
    **kwargs,  # Passed as extra params
):
    test_scores, train_scores = [], []
    scorer = get_scorer(scoring)
    est = clone(est).set_params(**kwargs)
    for step, (i0, i1) in enumerate(cv.split(X)):
        X0, y0, w0 = idx(X, i0), idx(y, i0), None if w is None else idx(w, i0)
        X1, y1, w1 = idx(X, i1), idx(y, i1), None if w is None else idx(w, i1)
        est.fit(X0, y0, sample_weight=w0)
        test_scores.append(scorer(est, X1, y1, sample_weight=w1))
        logger.info("Step %s test score = %s", step, test_scores[-1])
        if score_train:
            train_scores.append(scorer(est, X0, y0, sample_weight=w0))
            logger.info("Step %s train score = %s", step, train_scores[-1])
        if trial is not None:
            trial.report(np.mean(test_scores), step)
            if trial.should_prune():
                logger.info("Trial %s pruned at step %s", trial.number, step)
                raise optuna.TrialPruned()
    res = Bundle(
        test_score_mean=np.mean(test_scores),
        test_score_std=np.std(test_scores),
    )
    if score_train:
        res.train_score_mean = np.mean(train_scores)
        res.train_score_std = (np.std(train_scores),)
    return res


def lgbm_cv(
    params,
    X,
    y,
    w=None,
    cv=default_cv,
    scoring="r2",
    n_iters=100,
    log=False,
    early_stop=True,
    metrics=(),
    score_train=False,
    trial=None,
    **kwargs,  # Passed as extra params
):
    class PruningCallback:
        def __call__(self, env):
            for t in env.evaluation_result_list:
                if t[0] == "valid" and t[1] == "score":
                    trial.report(t[2], step=env.iteration)
            if trial.should_prune():
                logger.info("Trial %s pruned at step %s", trial.number, env.interation)
                raise optuna.TrialPruned()
            return False

    def metric(y_pred, data):
        y_real, weight = data.get_label(), data.get_weight()
        score = scorer._score_func(y_real, y_pred, sample_weight=weight)
        return "score", score, scorer._sign > 0

    params = params | kwargs
    n_iters = params.pop("n_estimators", n_iters)
    scorer = get_scorer(scoring)
    callbacks = []
    if trial is not None:
        callbacks.append(PruningCallback())
    if early_stop:
        cb_kwargs = dict(stopping_rounds=20, verbose=False)
        cb_kwargs.update(early_stop if isinstance(early_stop, dict) else {})
        callbacks.append(lgb.early_stopping(**cb_kwargs))
    if log:
        cb_kwargs = dict(show_stdv=True)
        cb_kwargs.update(log if isinstance(log, dict) else {})
        callbacks.append(lgb.log_evaluation(**cb_kwargs))
    info = lgb.cv(
        params=params,
        train_set=lgb.Dataset(X, label=y, weight=w, categorical_feature="auto"),
        num_boost_round=n_iters,
        folds=cv.split(X),
        feval=metric,
        metrics=metrics,
        eval_train_metric=score_train,
        callbacks=callbacks,
    )
    info = pd.DataFrame(info).rename(
        columns=lambda s: replace(
            s, {"-": "_", " ": "_", "valid": "test", "stdv": "std"}
        )
    )
    return Bundle(**info.iloc[-1], n_iters=len(info), info=info)


def xgb_cv(
    params,
    X,
    y,
    w=None,
    cv=default_cv,
    scoring="r2",
    n_iters=100,
    log=False,
    early_stop=True,
    metrics=(),
    trial=None,
    **kwargs,  # Passed as extra params
):
    class PruningCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            trial.report(evals_log["test"]["score"][-1][0], step=epoch)
            if trial.should_prune():
                logger.info("Trial %s pruned at step %s", trial.number, epoch)
                raise optuna.TrialPruned()
            return False

    def metric(y_pred, data):
        y_real, weight = data.get_label(), data.get_weight()
        score = scorer._score_func(y_real, y_pred, sample_weight=weight)
        return "score", score

    params = params | kwargs
    n_iters = params.pop("n_estimators", n_iters)
    scorer = get_scorer(scoring)
    callbacks = []
    if trial is not None:
        callbacks.append(PruningCallback())
    if early_stop:
        cb_kwargs = dict(rounds=20, maximize=scorer._sign > 0)
        cb_kwargs.update(early_stop if isinstance(early_stop, dict) else {})
        callbacks.append(xgb.callback.EarlyStopping(**cb_kwargs))
    if log:
        cb_kwargs = dict(show_stdv=True)
        cb_kwargs.update(log if isinstance(log, dict) else {})
        callbacks.append(xgb.callback.EvaluationMonitor(**cb_kwargs))
    info = xgb.cv(
        params=params,
        dtrain=xgb.DMatrix(X, label=y, weight=w, enable_categorical=True),
        num_boost_round=n_iters,
        folds=list(cv.split(X)),
        custom_metric=metric,
        metrics=metrics,
        callbacks=callbacks,
    ).rename(columns=lambda name: name.replace("-", "_"))
    return Bundle(**info.iloc[-1], n_iters=len(info), info=info)


def search(
    evaluate,
    space,
    n_trials=20,
    n_jobs=1,
    scoring="r2",
    score_penalty=1,
    name=None,
    storage=None,
    resume=False,
    sampler={},
    pruner={},
    verbosity="INFO",
    init_worker=lambda: None,
    seed=0,
    **kwargs,  # Passed to evaluate
):
    def suggest(trial):
        params = Bundle()
        for name, args in space.items():
            if type(args) is list:
                params[name] = trial.suggest_categorical(name, args)
            else:
                kind, *args = args
                suggest = {"f": trial.suggest_float, "i": trial.suggest_int}[kind]
                args = (args[:-1], args[-1]) if type(args[-1]) is dict else (args, {})
                params[name] = suggest(name, *args[0], **args[1])
        return params

    def objective(trial):
        logger.info("Running trial %s", trial.number)
        res = evaluate(suggest(trial), scoring, trial, **kwargs)
        for name, value in res.items():
            if name != "info":
                trial.set_user_attr(name, value)
        score = res.test_score_mean - scorer_sign * score_penalty * res.test_score_std
        logger.info("Trial %s ended with score %s", trial.number, score)
        return score

    def worker(id, n_trials):
        # https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
        # https://towardsdatascience.com/building-a-tree-structured-parzen-estimator-from-scratch-kind-of-20ed31770478/
        sampler_class = getattr(optuna.samplers, sampler.pop("name", "TPESampler"))
        pruner_class = getattr(optuna.pruners, pruner.pop("name", "MedianPruner"))
        optuna.load_study(
            study_name=name,
            storage=get_storage(storage),
            sampler=sampler_class(seed=rand(f"sampler-{id}", seed), **sampler),
            pruner=pruner_class(**pruner),
        ).optimize(objective, n_trials=n_trials)

    optuna.logging.set_verbosity(getattr(optuna.logging, verbosity))
    scorer_sign = get_scorer(scoring)._sign
    study = optuna.create_study(
        study_name=name,
        storage=get_storage(storage),
        direction="maximize" if scorer_sign > 0 else "minimize",
        load_if_exists=resume,
    )
    name = study.study_name
    logger.info("Running study '%s'", name)
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    n_trials = [n_trials // n_jobs + (i < n_trials % n_jobs) for i in range(n_jobs)]
    Parallel(n_jobs=n_jobs, initializer=init_worker)(
        delayed(worker)(i, n) for i, n in enumerate(n_trials) if n > 0
    )
    logger.info("Study '%s' finalized with best score %s", name, study.best_value)
    return study


def get_storage(storage=None, **kwargs):
    storage = storage or default_storage
    if not re.match(r"^[a-zA-Z]+://", storage):
        backend = optuna.storages.journal.JournalFileBackend(storage)
        storage = optuna.storages.JournalStorage(backend, **kwargs)
    else:
        storage = optuna.storages.RDBStorage(storage, **kwargs)
    return storage


def get_study(name, storage=None, **kwargs):
    return optuna.load_study(study_name=name, storage=get_storage(storage), **kwargs)


def get_importance(est, kind="gain"):
    if hasattr(est, "booster_"):  # LightGBM
        imp = pd.Series(
            est.booster_.feature_importance(importance_type=kind),
            index=est.booster_.feature_name(),
        )
    else:  # XGBoost
        imp = pd.Series(est.get_booster().get_score(importance_type=kind))
    return imp.sort_values(ascending=False)


def plot_importance(
    est,
    kind="gain",
    offset=0,
    limit=15,
    title="Feature importances",
    size=(6, 4),
    **kwargs,
):
    kinds = as_many(kind)
    n = len(kinds)
    n_rows = n // 2 + n % 2
    n_cols = min(n, 2)
    size = size[0] * n_cols, size[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=size)
    axes = axes.flatten() if n > 1 else [axes]
    axes[-1].axis("off" if 0 < n % 2 < n_cols else "on")
    for kind, ax in zip(kinds, axes):
        imp = get_importance(est, kind).iloc[offset:limit]
        imp.plot(kind="barh", ax=ax, **kwargs).invert_yaxis()
    fig.suptitle(title)
    plt.tight_layout()
