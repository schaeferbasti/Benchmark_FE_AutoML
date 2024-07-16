from amltk import Trial, Node
from amltk.sklearn import CVEvaluation


def get_cv_evaluator(X, y, X_test, y_test, inner_fold_seed, on_trial_exception, task_hint,):
    return CVEvaluation(
        # Provide data, number of times to split, cross-validation and a hint of the task type
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        splitter="cv",
        n_splits=8,
        task_hint=task_hint,
        # Seeding for reproducibility
        random_state=inner_fold_seed,
        # Record training scores
        train_score=True,
        # Where to store things
        working_dir="logs/log.txt",
        # What to do when something goes wrong.
        on_error="raise" if on_trial_exception == "raise" else "fail",
        # Whether you want models to be store on disk under working_dir
        store_models=False,
        # A callback to be called at the end of each split
        post_split=do_something_after_a_split_was_evaluated,
        # Some callback that is called at the end of all fold evaluations
        post_processing=do_something_after_a_complete_trial_was_evaluated,
        # Whether the post_processing callback requires models will require models, i.e.
        # to compute some bagged average over all fold models. If `False` will discard models eagerly
        # to save space.
        post_processing_requires_models=False,
        # This handles edge cases related to stratified splitting when there are too
        # few instances of a specific class. May wish to disable if your passing extra fit params
        # rebalance_if_required_for_stratified_splitting=True,
        # Extra parameters requested by sklearn models/group splitters or metrics,
        # such as `sample_weight`
        params=None,
    )

def do_something_after_a_split_was_evaluated(
        trial: Trial,
        fold: int,
        info: CVEvaluation.PostSplitInfo,
) -> CVEvaluation.PostSplitInfo:
    return info


def do_something_after_a_complete_trial_was_evaluated(
        report: Trial.Report,
        pipeline: Node,
        info: CVEvaluation.CompleteEvalInfo,
) -> Trial.Report:
    return report