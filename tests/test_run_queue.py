"""scripts/run_queue.sh against stub nvidia-smi, python, swift and train.sh. Touches no GPU."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import common  # noqa: E402


def stub(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body + "\n")
    path.chmod(0o755)


def make_repo(tmp_path: Path, snapshots: list[str], fail_on: str = "") -> Path:
    """A repo copy where every command appends '<VQA_GPU>|<args>' to calls.txt. The last snapshot repeats."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/run_queue.sh", repo / "scripts/run_queue.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for i, snapshot in enumerate(snapshots):
        (tmp_path / f"snapshot{i}").write_text(snapshot)
    stub(bin_dir / "nvidia-smi", f"""
count=$(cat "{tmp_path}/count" 2>/dev/null || echo 0)
echo $((count + 1)) > "{tmp_path}/count"
((count < {len(snapshots) - 1})) || count={len(snapshots) - 1}
cat "{tmp_path}/snapshot$count"
""")
    stub(bin_dir / "swift", "exit 0")
    record = f"""
[[ "$*" == *-c\\ import\\ torch* || "$1" == "-" ]] && exit 0
cat > /dev/null  # a step that reads stdin must not swallow the remaining steps
echo "${{VQA_GPU:-none}}|$*" >> "{tmp_path}/calls.txt"
[[ -n "{fail_on}" && "$*" == *"{fail_on}"* && ! -e "{tmp_path}/fixed" ]] && exit 7
exit 0
"""
    stub(bin_dir / "python", record)
    stub(repo / "scripts/train.sh", record)
    return repo


def run_queue(repo: Path, tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    env.pop("VQA_GPU", None)
    env.pop("PYTHON", None)
    return subprocess.run(["bash", str(repo / "scripts/run_queue.sh"), *args, "--poll-seconds", "0", "--stable-polls", "2"],
                          env=env, capture_output=True, text=True, timeout=60)


def calls(tmp_path: Path) -> list[str]:
    return (tmp_path / "calls.txt").read_text().splitlines()


BUSY = "0, 23222, 100\n1, 23222, 100\n2, 23922, 100\n"
GPU1_FREE = "0, 23222, 100\n1, 10, 0\n2, 23922, 100\n"


def test_waits_for_stable_free_gpu_and_runs_steps_in_order(tmp_path):
    # GPU 1 is free for one check, busy again, then stays free: the blip must not count.
    repo = make_repo(tmp_path, [BUSY, GPU1_FREE, BUSY, GPU1_FREE])
    result = run_queue(repo, tmp_path, "smoke")
    assert result.returncode == 0, result.stdout + result.stderr
    assert int((tmp_path / "count").read_text()) >= 5  # four snapshots plus the stable confirmation
    assert calls(tmp_path) == [
        "none|e1_t1 --dry-run",
        "1|e1_t1 --smoke --force",
        "1|src/evaluate.py --experiment e0 --limit 5 --overwrite",
        "1|src/evaluate.py --experiment e1_t1 --limit 5 --overwrite",
        "none|src/report.py --estimate --hours-available 24",
    ]


def test_only_allowed_gpus_are_used(tmp_path):
    both_free = "0, 10, 0\n1, 23222, 100\n2, 10, 0\n"
    repo = make_repo(tmp_path, [both_free])
    assert run_queue(repo, tmp_path, "mandatory", "--gpus", "2").returncode == 0
    gpu_calls = [c for c in calls(tmp_path) if not c.startswith("none|")]
    assert gpu_calls and all(c.startswith("2|") for c in gpu_calls)


def test_unknown_gpu_is_an_error_not_an_endless_wait(tmp_path):
    repo = make_repo(tmp_path, [BUSY])
    result = run_queue(repo, tmp_path, "mandatory", "--gpus", "9")
    assert result.returncode == 2 and "No GPU matches" in result.stdout  # stderr goes through the queue log


def test_failure_stops_queue_and_rerun_resumes_at_failed_step(tmp_path):
    repo = make_repo(tmp_path, [GPU1_FREE], fail_on="e1_t2")
    result = run_queue(repo, tmp_path, "mandatory")
    assert result.returncode == 7
    assert calls(tmp_path)[-1] == "1|e1_t2"
    assert not any("e1_t3" in c for c in calls(tmp_path))

    (tmp_path / "fixed").touch()
    (tmp_path / "calls.txt").unlink()
    result = run_queue(repo, tmp_path, "mandatory")
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls(tmp_path)[0] == "1|e1_t2"
    assert not any(c == "1|e1_t1" for c in calls(tmp_path))


def test_report_runs_again_in_each_stage(tmp_path):
    repo = make_repo(tmp_path, [GPU1_FREE])
    assert run_queue(repo, tmp_path, "mandatory", "exploratory").returncode == 0
    assert calls(tmp_path).count("none|src/report.py") == 2


def test_missing_swift_fails_before_waiting(tmp_path):
    repo = make_repo(tmp_path, [BUSY])
    (tmp_path / "bin/swift").unlink()
    path = f"{tmp_path / 'bin'}:/usr/bin:/bin"
    if shutil.which("swift", path=path):
        return
    result = subprocess.run(["bash", str(repo / "scripts/run_queue.sh"), "smoke"], env={**os.environ, "PATH": path},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and "conda activate datpt_ml" in result.stderr
    assert not (tmp_path / "count").exists()


def test_dry_run_touches_nothing(tmp_path):
    repo = make_repo(tmp_path, [BUSY])
    result = run_queue(repo, tmp_path, "smoke", "--dry-run")
    assert result.returncode == 0 and "smoke|gpu|bash scripts/train.sh e1_t1 --smoke --force" in result.stdout
    assert not (tmp_path / "count").exists() and not (repo / "runs").exists()


def test_vqa_gpu_overrides_config_gpu(monkeypatch):
    monkeypatch.delenv("VQA_GPU", raising=False)
    assert common.load_config()["gpu"] == 0
    monkeypatch.setenv("VQA_GPU", "2")
    assert common.load_config()["gpu"] == 2


def test_v2_stages_name_their_config_and_keep_seeds_apart(tmp_path):
    repo = make_repo(tmp_path, [GPU1_FREE])
    result = run_queue(repo, tmp_path, "v2", "v2-seeds", "--gpus", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    steps = calls(tmp_path)
    assert all(c.startswith("1|") or c.startswith("none|src/report.py") for c in steps)
    assert all("--config config.v2_1000train_500test.yaml" in c for c in steps)
    assert steps[:3] == ["1|src/evaluate.py --config config.v2_1000train_500test.yaml --experiment e0",
                         "1|src/evaluate.py --config config.v2_1000train_500test.yaml --experiment e3 --adapters-from v1_400train",
                         "1|src/evaluate.py --config config.v2_1000train_500test.yaml --experiment e4 --adapters-from v1_400train"]
    for seed in (42, 43, 44):
        for experiment in ("e1_t1", "e1_t4", "e2_joint", "e3_query", "e4_relate"):
            assert sum(c.startswith(f"1|{experiment} ") and f"--seed {seed}" in c for c in steps) == 1
        assert sum(f"--experiment e1 --seed {seed}" in c for c in steps) == 1
    done = (repo / "runs/queue/done.txt").read_text()
    assert "mandatory:" not in done and len(set(done.splitlines())) == len(done.splitlines())
