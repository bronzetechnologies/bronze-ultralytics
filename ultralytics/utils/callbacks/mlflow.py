# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MLflow Logging for Ultralytics YOLO.

This module enables MLflow logging for Ultralytics YOLO. It logs metrics, parameters, and model artifacts.
For setting up, a tracking URI should be specified. The logging can be customized using environment variables.

Commands:
    1. To set a project name:
        `export MLFLOW_EXPERIMENT_NAME=<your_experiment_name>` or use the project=<project> argument

    2. To set a run name:
        `export MLFLOW_RUN=<your_run_name>` or use the name=<name> argument

    3. To start a local MLflow server:
        mlflow server --backend-store-uri runs/mlflow
       It will by default start a local server at http://127.0.0.1:5000.
       To specify a different URI, set the MLFLOW_TRACKING_URI environment variable.

    4. To kill all running MLflow server instances:
        ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
"""

from ultralytics.utils import LOGGER, RUNS_DIR, SETTINGS, TESTS_RUNNING, colorstr

# Bronze: every save_dir result file goes to this run folder, not to the run root
RESULT_ARTIFACTS_DIR = "result artifacts"

# Bronze: an MLflow outage must not kill a long training. Helpers shared with engine/trainer.py.
BEST_EFFORT_COOLDOWN_S = 180
FATAL_PATIENCE_S = 1800
FATAL_SLEEP_S = 60
_PROGRAMMING_ERRORS = (TypeError, AttributeError, NameError)


def _mlflow_client(trainer):
    """Return the duck-typed Bronze MLflow client of the trainer, or None."""
    return getattr(getattr(trainer, "pipe_logger", None), "mlflow_client", None)


def _best_effort(trainer, what, fn, *args, **kwargs) -> bool:
    """Run fn(*args, **kwargs) without ever raising on an MLflow outage. Return True on success."""
    import os
    import time

    client = _mlflow_client(trainer)
    if client is not None and hasattr(client, "best_effort"):
        return client.best_effort(what, fn, *args, **kwargs)
    # Fallback: minimal local circuit breaker
    now = time.monotonic()
    skip_until = getattr(trainer, "_mlflow_skip_until", 0.0)
    if now < skip_until:
        trainer._mlflow_best_effort_skipped = getattr(trainer, "_mlflow_best_effort_skipped", 0) + 1
        return False
    key = "MLFLOW_HTTP_REQUEST_MAX_RETRIES"
    old = os.environ.get(key)
    os.environ[key] = "2"
    try:
        fn(*args, **kwargs)
        return True
    except _PROGRAMMING_ERRORS:
        raise
    except Exception as e:
        trainer._mlflow_best_effort_failures = getattr(trainer, "_mlflow_best_effort_failures", 0) + 1
        trainer._mlflow_skip_until = time.monotonic() + BEST_EFFORT_COOLDOWN_S
        LOGGER.warning(f"MLflow: {what} failed, training continues ({type(e).__name__}: {e})")
        return False
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _reset_best_effort(trainer):
    """End the best-effort cooldown when the epochs end, so the end-of-training uploads are tried at least once."""
    client = _mlflow_client(trainer)
    if client is not None and hasattr(client, "reset_best_effort_cooldown"):
        client.reset_best_effort_cooldown()
    trainer._mlflow_skip_until = 0.0


def _fatal(trainer, what, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), retrying for up to FATAL_PATIENCE_S, then raise."""
    import time

    client = _mlflow_client(trainer)
    if client is not None and hasattr(client, "fatal_with_patience"):
        return client.fatal_with_patience(what, fn, *args, **kwargs)
    deadline = time.monotonic() + FATAL_PATIENCE_S
    while True:
        try:
            return fn(*args, **kwargs)
        except _PROGRAMMING_ERRORS:
            raise
        except Exception as e:
            if time.monotonic() >= deadline:
                raise
            LOGGER.warning(f"MLflow: {what} failed, retrying in {FATAL_SLEEP_S}s ({type(e).__name__}: {e})")
            time.sleep(FATAL_SLEEP_S)


try:
    import os

    assert not TESTS_RUNNING or "test_mlflow" in os.environ.get("PYTEST_CURRENT_TEST", "")  # do not log pytest
    assert SETTINGS["mlflow"] is True  # verify integration is enabled
    import mlflow

    assert hasattr(mlflow, "__version__")  # verify package is not directory
    from pathlib import Path

    PREFIX = colorstr("MLflow: ")

except (ImportError, AssertionError):
    mlflow = None


def sanitize_dict(x: dict) -> dict:
    """Sanitize dictionary keys by removing parentheses and converting values to floats."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}


def on_pretrain_routine_end(trainer):
    """
    Log training parameters to MLflow at the end of the pretraining routine.

    This function sets up MLflow logging based on environment variables and trainer arguments. It sets the tracking URI,
    experiment name, and run name, then starts the MLflow run if not already active. It finally logs the parameters
    from the trainer.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The training object with arguments and parameters to log.

    Environment Variables:
        MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
        MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
        MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
        MLFLOW_KEEP_RUN_ACTIVE: Boolean indicating whether to keep the MLflow run active after training ends.
    """
    global mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(RUNS_DIR / "mlflow")
    LOGGER.debug(f"{PREFIX} tracking uri: {uri}")
    mlflow.set_tracking_uri(uri)

    # Set experiment and run names
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or trainer.args.project or "/Shared/Ultralytics"
    run_name = os.environ.get("MLFLOW_RUN") or trainer.args.name
    mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    try:
        active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
        LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
        if Path(uri).is_dir():
            LOGGER.info(f"{PREFIX}view at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri {uri}'")
        LOGGER.info(f"{PREFIX}disable with 'yolo settings mlflow=False'")
        mlflow.log_params(dict(trainer.args))
    except Exception as e:
        LOGGER.warning(f"{PREFIX}Failed to initialize: {e}")
        LOGGER.warning(f"{PREFIX}Not tracking this run")


def on_train_epoch_end(trainer):
    """Log training metrics at the end of each train epoch to MLflow."""
    if mlflow:
        _best_effort(
            trainer,
            "train epoch metrics",
            lambda: mlflow.log_metrics(
                metrics={
                    **sanitize_dict(trainer.lr),
                    **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
                },
                step=trainer.epoch,
            ),
        )


def on_fit_epoch_end(trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        _best_effort(
            trainer,
            "fit epoch metrics",
            lambda: mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=trainer.epoch),
        )


def on_train_end(trainer):
    """Log model artifacts at the end of training."""
    if not mlflow:
        return
    # Bronze: best.pt is fatal (retried with patience); last.pt and other files are best-effort.
    # Same artifact paths as logging the whole weights/ directory: weights/<file>
    weights_dir = trainer.best.parent
    best_name = Path(trainer.best).name
    for f in sorted(weights_dir.glob("*")):
        if not f.is_file():
            continue
        if f.name == best_name:
            _fatal(trainer, f"upload weights/{f.name}", mlflow.log_artifact, str(f), weights_dir.name)
        else:
            _best_effort(trainer, f"upload weights/{f.name}", mlflow.log_artifact, str(f), weights_dir.name)
    for f in trainer.save_dir.glob("*"):  # log all other files in save_dir, in one folder (Bronze)
        if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
            _best_effort(trainer, f"upload {f.name}", mlflow.log_artifact, str(f), RESULT_ARTIFACTS_DIR)
    if _mlflow_client(trainer) is None:
        n = getattr(trainer, "_mlflow_best_effort_failures", 0)
        k = getattr(trainer, "_mlflow_best_effort_skipped", 0)
        LOGGER.info(f"{PREFIX}best-effort MLflow calls failed: {n}, skipped during cooldown: {k}")
    keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
    if keep_run_active:
        LOGGER.info(f"{PREFIX}mlflow run still alive, remember to close it using mlflow.end_run()")
    else:
        mlflow.end_run()
        LOGGER.debug(f"{PREFIX}mlflow run ended")

    LOGGER.info(
        f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n{PREFIX}disable with 'yolo settings mlflow=False'"
    )


callbacks = (
    {
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if mlflow
    else {}
)
