import logging
import re
import tempfile

import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn import set_config
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold

from .base import try_import

lgb = try_import("lightgbm")
xgb = try_import("xgboost")
optuna = try_import("optuna")

logger = logging.getLogger(__name__)


def config_sklearn(pandas_output=True):
    if pandas_output:
        # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html
        set_config(transform_output="pandas")


def search(
    new_est,
    space,
    X,
    y,
    w=None,
    *,
    cv=5,
    scoring="r2",
    score_penalty=0,
    score_train=False,
    early_stop=False,
    n_trials=20,
    n_jobs=-1,
    name=None,
    storage=None,
    resume=False,
    sampler_kwargs={},
    pruner_kwargs={},
    verbosity="WARNING",
    seed=0,
    **kwargs
):
    def suggest(trial):
        params = {}
        for name, args in space.items():
            if type(args) is list:
                params[name] = trial.suggest_categorical(name, args)
            else:
                name, kind = name.rsplit(":", 1)
                suggest = {"f": trial.suggest_float, "i": trial.suggest_int}[kind]
                args = (args[:-1], args[-1]) if type(args[-1]) is dict else (args, {})
                params[name] = suggest(name, *args[0], **args[1])
        return params

    def objective(trial):
        logger.info("Running trial %s", trial.number)
        scores, train_scores, best_iterations = [], [], []
        est = new_est(**suggest(trial), refit=False, **kwargs)
        for step, (i0, i1) in enumerate(cv.split(X)):
            X0, y0, w0 = X.iloc[i0], y.iloc[i0], None if w is None else w.iloc[i0]
            X1, y1, w1 = X.iloc[i1], y.iloc[i1], None if w is None else w.iloc[i1]
            fit_kwargs = {}
            if early_stop not in (None, False):
                fit_kwargs["eval"] = X1, y1, w1, scorer
                fit_kwargs.update({} if early_stop is True else early_stop)
            est.fit(X0, y0, sample_weight=w0, **fit_kwargs)
            scores.append(scorer(est, X1, y1, sample_weight=w1))
            if score_train:
                train_scores.append(scorer(est, X0, y0, sample_weight=w0))
            if early_stop:
                best_iterations.append(est.best_iteration_)
            trial.report(np.mean(scores), step)
            if trial.should_prune():
                logger.info("Trial %s prunned", trial.number)
                raise optuna.TrialPruned()
        trial.set_user_attr("score", np.mean(scores))
        trial.set_user_attr("std_score", np.std(scores))
        if score_train:
            trial.set_user_attr("train_score", np.mean(train_scores))
            trial.set_user_attr("train_std_score", np.std(train_scores))
        if early_stop:
            trial.set_user_attr("best_iterations", best_iterations)
        score = np.mean(scores) - score_penalty * np.std(scores)
        logger.info("Trial %s ended with score %s", trial.number, score)
        return score

    def worker(id, n_trials):
        # https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
        # https://towardsdatascience.com/building-a-tree-structured-parzen-estimator-from-scratch-kind-of-20ed31770478/
        get_study(
            name,
            storage,
            sampler=optuna.samplers.TPESampler(seed=seed + id, **sampler_kwargs),
            pruner=optuna.pruners.MedianPruner(**pruner_kwargs),
            load=True,
        ).optimize(objective, n_trials=n_trials)

    logger.info("Running study '%s'", name)
    cv = KFold(cv) if type(cv) is int else cv
    scorer = get_scorer(scoring)
    optuna.logging.set_verbosity(getattr(optuna.logging, verbosity))
    with tempfile.NamedTemporaryFile() as f:  # Ignored if storage is passed
        storage = storage or f.name
        study = get_study(name, storage, direction="maximize", load=resume)
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        n_trials = [n_trials // n_jobs + (i < n_trials % n_jobs) for i in range(n_jobs)]
        Parallel(n_jobs=n_jobs)(
            delayed(worker)(i, n) for i, n in enumerate(n_trials) if n > 0
        )
        logger.info("Refitting estimator for study '%s'", name)
        est = new_est(**study.best_params, **kwargs, refit=True)
        if early_stop:
            est.set_best_iteration(study.best_trial.user_attrs["best_iterations"])
        est.fit(X, y, sample_weight=w)

        return study, est


def get_study(name, storage, load=True, **kwargs):
    if not re.match(r"^[a-zA-Z]+://", storage):
        backend = optuna.storages.journal.JournalFileBackend(storage)
        storage = optuna.storages.JournalStorage(backend)
    return optuna.create_study(
        study_name=name, storage=storage, load_if_exists=load, **kwargs
    )


class BaseEarlyStoppingMixin:
    def set_best_iteration(self, best_iterations):
        self.set_params(n_estimators=round(np.mean(best_iterations)))


class LGBMEarlyStoppingMixin(BaseEarlyStoppingMixin):
    def fit(self, *args, eval=None, rounds=20, tol=0, verbose=False, **kwargs):
        if not eval:
            return super().fit(*args, **kwargs)

        def metric(y_true, y_pred, weight):
            score = scorer._score_func(y_true, y_pred, sample_weight=weight)
            return "score", score, scorer._sign > 0

        X, y, w, scorer = eval
        cb = lgb.early_stopping(stopping_rounds=rounds, min_delta=tol, verbose=verbose)
        return super().fit(
            *args,
            eval_set=[(X, y)],
            eval_sample_weight=[w],
            eval_metric=metric,
            callbacks=[cb],
            **kwargs
        )


class XGBEarlyStoppingMixin(BaseEarlyStoppingMixin):

    def fit(self, *args, eval=None, rounds=20, tol=0, verbose=False, **kwargs):
        if not eval:
            return super().fit(*args, **kwargs)

        # https://github.com/dmlc/xgboost/discussions/12040#discussioncomment-15949111
        X, y, w, scorer = eval
        cb = xgb.callback.EarlyStopping(
            rounds=rounds, min_delta=tol, maximize=scorer._sign > 0, save_best=True
        )
        self.set_params(eval_metric=scorer._score_func, callbacks=[cb])
        return super().fit(
            *args,
            eval_set=[(X, y)],
            sample_weight_eval_set=[w],
            verbose=verbose,
            **kwargs
        )

    @property
    def best_iteration_(self):
        return self.best_iteration
