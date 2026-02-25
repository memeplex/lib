import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn import set_config
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold

from .base import try_import

lgb = try_import("lightgbm")
optuna, ostore = try_import("optuna"), try_import("optuna.storages")


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
    sampler_kwargs={},
    pruner_kwargs={},
    verbosity="WARNING",
    seed=0,
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
        scores, train_scores, stopped_at = [], [], []
        est = new_est(**suggest(trial), refit=False)
        for step, (i0, i1) in enumerate(cv.split(X)):
            X0, y0, w0 = X.iloc[i0], y.iloc[i0], None if w is None else w.iloc[i0]
            X1, y1, w1 = X.iloc[i1], y.iloc[i1], None if w is None else w.iloc[i1]
            kwargs = {}
            if early_stop not in (None, False):
                kwargs["eval"] = X1, y1, w1, scorer
                kwargs.update({} if early_stop is True else early_stop)
            est.fit(X0, y0, sample_weight=w0, **kwargs)
            scores.append(scorer(est, X1, y1, sample_weight=w1))
            if score_train:
                train_scores.append(scorer(est, X0, y0, sample_weight=w0))
            if early_stop:
                stopped_at.append(est.stopped_at)
            trial.report(np.mean(scores), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("score", np.mean(scores))
        trial.set_user_attr("std_score", np.std(scores))
        if score_train:
            trial.set_user_attr("train_score", np.mean(train_scores))
            trial.set_user_attr("train_std_score", np.std(train_scores))
        if early_stop:
            trial.set_user_attr("stopped_at", stopped_at)
        return np.mean(scores) - score_penalty * np.std(scores)

    def worker(n_trials):
        study = optuna.load_study(study_name=name, storage=storage)
        study.optimize(objective, n_trials=n_trials)

    cv = KFold(cv) if type(cv) is int else cv
    scorer = get_scorer(scoring)
    optuna.logging.set_verbosity(getattr(optuna.logging, verbosity))
    if type(storage) is str and storage[0] in (".", "/"):
        storage = ostore.JournalStorage(ostore.journal.JournalFileBackend(storage))
    # https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
    # https://towardsdatascience.com/building-a-tree-structured-parzen-estimator-from-scratch-kind-of-20ed31770478/
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=seed, **sampler_kwargs),
        pruner=optuna.pruners.MedianPruner(**pruner_kwargs),
        direction="maximize",
    )
    if storage is None:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    else:
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        n_trials = [n_trials // n_jobs + (i < n_trials % n_jobs) for i in range(n_jobs)]
        Parallel(n_jobs=n_jobs)(delayed(worker)(n) for n in n_trials if n > 0)
    est = new_est(**study.best_params, refit=True)
    if early_stop:
        est.stop_at(study.best_trial.user_attrs["stopped_at"])
    est.fit(X, y, sample_weight=w)

    return study, est


class LGBMEarlyStoppingMixin:
    def fit(self, *args, eval=None, rounds=20, verbose=False, **kwargs):
        def metric(y_true, y_pred, weight):
            score = scorer._score_func(y_true, y_pred, sample_weight=weight)
            return "score", score, scorer._sign > 0

        if eval:
            X, y, w, scorer = eval
            kwargs.update(
                eval_set=[(X, y)],
                eval_sample_weight=[w],
                eval_metric=metric,
                callbacks=[lgb.early_stopping(stopping_rounds=rounds, verbose=verbose)],
            )
        return super().fit(*args, **kwargs)

    def stop_at(self, stopped_at):
        self.set_params(n_estimators=round(np.mean(stopped_at)))

    @property
    def stopped_at(self):
        return self.best_iteration_
