import numpy as np
from sklearn import set_config
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold


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
    n_trials=20,
    early=False,
    train_score=False,
    n_jobs=-1,
    verbosity="WARNING",
    sampler_kwargs={},
    pruner_kwargs={},
    seed=0,
    storage=None,
):
    import optuna

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
        scores, train_scores = [], []
        est = new_est(**suggest(trial))
        for step, (i0, i1) in enumerate(cv.split(X)):
            X0, y0, w0 = X.iloc[i0], y.iloc[i0], None if w is None else w.iloc[i0]
            X1, y1, w1 = X.iloc[i1], y.iloc[i1], None if w is None else w.iloc[i1]
            eval_kwargs = {"eval": (X1, y1, w1, scorer)} if early else {}
            est.fit(X0, y0, sample_weight=w0, **eval_kwargs)
            scores.append(scorer(est, X1, y1, sample_weight=w1))
            if train_score:
                train_scores.append(scorer(est, X0, y0, sample_weight=w0))
            trial.report(np.mean(scores), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("std_score", np.std(scores))
        if train_score:
            trial.set_user_attr("train_score", np.mean(train_scores))
            trial.set_user_attr("train_std_score", np.std(train_scores))
        return np.mean(scores)

    cv = KFold(cv) if type(cv) is int else cv
    scorer = get_scorer(scoring)
    optuna.logging.set_verbosity(getattr(optuna.logging, verbosity))
    # https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
    # https://towardsdatascience.com/building-a-tree-structured-parzen-estimator-from-scratch-kind-of-20ed31770478/
    sampler = optuna.samplers.TPESampler(seed=seed, **sampler_kwargs)
    pruner = optuna.pruners.MedianPruner(**pruner_kwargs)
    study = optuna.create_study(
        sampler=sampler, pruner=pruner, storage=storage, direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    est = new_est(**study.best_params).fit(X, y, sample_weight=w)

    return study, est
