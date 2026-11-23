"""Bronze: an MLflow outage must not kill a training. Fakes only, no network, no real training."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.callbacks import mlflow as cb


class StubClient:
    def __init__(self, ok=True):
        self.ok = ok
        self.model_calls = []
        self.artifact_calls = []

    def best_effort(self, what, fn, *a, **k):
        if not self.ok:
            return False
        fn(*a, **k)
        return True

    def fatal_with_patience(self, what, fn, *a, **k):
        return fn(*a, **k)

    def log_pt_model_mlflow(self, ckpt, backbone):
        self.model_calls.append((ckpt, backbone))

    def log_artifact(self, path, folder=None):
        self.artifact_calls.append(path)

    def log_state_dict_mlflow(self, ckpt, name):
        pass


class FakeMlflow:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.artifacts = []

    def log_metrics(self, metrics, step=None):
        if self.fail:
            raise ConnectionError("down")
        self.calls.append((metrics, step))

    def log_artifact(self, path, folder=None):
        if self.fail:
            raise ConnectionError("down")
        self.artifacts.append((Path(path).name, folder))

    def get_tracking_uri(self):
        return "x"

    def end_run(self):
        pass


def make_trainer(client=None, **kw):
    t = BaseTrainer.__new__(BaseTrainer)
    t.pipe_logger = SimpleNamespace(mlflow_client=client) if client is not None else None
    t._pending_best_ckpt = None
    t.__dict__.update(kw)
    return t


def epoch_trainer(client):
    return make_trainer(
        client,
        epoch=3,
        lr={"lr/pg0": 0.1},
        tloss=None,
        metrics={"metrics/mAP50(M)": 0.5},
        label_loss_items=lambda tloss, prefix="train": {"train/box_loss": 1.0},
    )


def test_epoch_metrics_do_not_raise_when_best_effort_false(monkeypatch):
    monkeypatch.setattr(cb, "mlflow", FakeMlflow())
    t = epoch_trainer(StubClient(ok=False))
    cb.on_train_epoch_end(t)
    cb.on_fit_epoch_end(t)


def test_epoch_metrics_do_not_raise_when_mlflow_raises_in_fallback(monkeypatch):
    monkeypatch.setattr(cb, "mlflow", FakeMlflow(fail=True))
    t = epoch_trainer(None)
    cb.on_train_epoch_end(t)
    cb.on_fit_epoch_end(t)
    assert t._mlflow_best_effort_failures == 1  # second call skipped by the cooldown
    assert t._mlflow_best_effort_skipped == 1


def test_val_metrics_one_call(monkeypatch):
    fm = FakeMlflow()
    monkeypatch.setattr(cb, "mlflow", fm)
    t = make_trainer(StubClient())
    t._log_metrics_best_effort("val", {"val/a": 1.0, "val/b": 2.0}, 4)
    assert fm.calls == [({"val/a": 1.0, "val/b": 2.0}, 4)]


def test_best_fails_then_later_best_succeeds_nothing_pending():
    client = StubClient()
    t = make_trainer(client)
    t._pending_best_ckpt = ("old", "yolo")
    # a later best replaces the pending one on success
    assert cb._best_effort(t, "m", client.log_pt_model_mlflow, "new", "yolo")
    t._pending_best_ckpt = None
    n = len(client.model_calls)
    t._ensure_best_model_uploaded()
    assert len(client.model_calls) == n


def test_last_best_failed_one_fatal_reupload():
    client = StubClient()
    t = make_trainer(client)
    t._pending_best_ckpt = ("ckpt", "yolo")
    t._ensure_best_model_uploaded()
    assert client.model_calls == [("ckpt", "yolo")]
    assert t._pending_best_ckpt is None
    t._ensure_best_model_uploaded()
    assert len(client.model_calls) == 1


def test_fatal_reupload_failure_propagates():
    client = StubClient()

    def boom(ckpt, backbone):
        raise ConnectionError("down")

    client.log_pt_model_mlflow = boom
    t = make_trainer(client)
    t._pending_best_ckpt = ("ckpt", "yolo")
    with pytest.raises(ConnectionError):
        t._ensure_best_model_uploaded()
    assert t._pending_best_ckpt is not None


def train_end_trainer(tmp_path, client=None):
    w = tmp_path / "weights"
    w.mkdir()
    (w / "best.pt").write_bytes(b"b")
    (w / "last.pt").write_bytes(b"l")
    (tmp_path / "results.csv").write_text("x")
    return make_trainer(client, best=w / "best.pt", save_dir=tmp_path)


class SelectiveMlflow(FakeMlflow):
    def __init__(self, failing):
        super().__init__()
        self.failing = failing

    def log_artifact(self, path, folder=None):
        if Path(path).name in self.failing:
            raise ConnectionError("down")
        super().log_artifact(path, folder)


def test_best_pt_failure_raises_last_pt_does_not(monkeypatch, tmp_path):
    monkeypatch.setattr(cb, "FATAL_PATIENCE_S", 0)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(cb, "PREFIX", "", raising=False)
    monkeypatch.setattr(cb, "Path", Path, raising=False)
    monkeypatch.setattr(cb, "mlflow", SelectiveMlflow({"best.pt"}))
    with pytest.raises(ConnectionError):
        cb.on_train_end(train_end_trainer(tmp_path))

    fm = SelectiveMlflow({"last.pt"})
    monkeypatch.setattr(cb, "mlflow", fm)
    other = tmp_path / "second"
    other.mkdir()
    cb.on_train_end(train_end_trainer(other))  # last.pt fails and is swallowed
    assert ("best.pt", "weights") in fm.artifacts
    assert not any(n == "last.pt" for n, _ in fm.artifacts)


def test_fallback_counts_and_restores_env(monkeypatch):
    t = make_trainer(None)
    seen = {}

    def fn():
        seen["v"] = os.environ.get("MLFLOW_HTTP_REQUEST_MAX_RETRIES")
        raise ConnectionError("down")

    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "7")
    assert cb._best_effort(t, "x", fn) is False
    assert seen["v"] == "2"
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "7"
    assert t._mlflow_best_effort_failures == 1
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES")
    t._mlflow_skip_until = 0
    cb._best_effort(t, "x", fn)
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ


def test_fallback_programming_error_reraised():
    t = make_trainer(None)
    with pytest.raises(TypeError):
        cb._best_effort(t, "x", lambda: (_ for _ in ()).throw(TypeError("bug")))


def test_failed_result_image_retried(tmp_path):
    (tmp_path / "a.png").write_bytes(b"1")
    client = StubClient(ok=False)
    t = make_trainer(client, save_dir=tmp_path)
    t._log_new_result_images()
    assert t._uploaded_result_images == set()
    client.ok = True
    t._log_new_result_images()
    assert t._uploaded_result_images == {"a.png"}
    assert len(client.artifact_calls) == 1


def test_pending_best_without_client_raises():
    t = make_trainer(None)
    t._pending_best_ckpt = ("ckpt", "yolo")
    with pytest.raises(RuntimeError):
        t._ensure_best_model_uploaded()
