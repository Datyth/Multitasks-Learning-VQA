"""Pipeline checks on synthetic annotations and tiny images. Loads no Qwen weights or real GQA data."""
import json
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import common  # noqa: E402
import evaluate  # noqa: E402
import prepare  # noqa: E402
import report  # noqa: E402

RULES = {"T1": ("query", False), "T2": ("query", True), "T3": ("verify", False), "T4": ("verify", True)}


def make_question(task, image_id, qid):
    structural, relate = RULES[task]
    ops = ["select"] + (["relate"] if relate else []) + ["query" if structural == "query" else "verify color"]
    return {
        "imageId": image_id,
        "question": f"Question {qid}?",
        "answer": ("yes" if int(qid) % 2 else "no") if structural == "verify" else f"thing{int(qid) % 5}",
        "isBalanced": True,
        "types": {"structural": structural, "semantic": "rel" if relate else "attr", "detailed": "d"},
        "semantic": [{"operation": op, "argument": "", "dependencies": []} for op in ops],
    }


def make_annotations(prefix, n_images, questions_per_task, start_qid):
    questions, qid = {}, start_qid
    for i in range(n_images):
        image_id = f"{prefix}{i}"
        for task in RULES:
            for _ in range(questions_per_task):
                questions[str(qid)] = make_question(task, image_id, qid)
                qid += 1
        questions[str(qid)] = {**make_question("T1", image_id, qid),
                               "types": {"structural": "choose", "semantic": "attr", "detailed": "d"}}
        qid += 1
    return questions


def jpeg_bytes():
    buffer = BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def workspace(tmp_path):
    return make_workspace(tmp_path)


def make_workspace(tmp_path, held_out_images=10):
    """Synthetic question and image archives plus a small config."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    train = make_annotations("n", 30, 2, 1000)
    held_out = make_annotations("v", held_out_images, 2, 900000)
    with zipfile.ZipFile(raw / "questions1.2.zip", "w") as zf:
        zf.writestr("train_balanced_questions.json", json.dumps(train))
        zf.writestr("val_balanced_questions.json", json.dumps(held_out))
    with zipfile.ZipFile(raw / "images.zip", "w") as zf:
        zf.writestr("images/", "")
        for image_id in {q["imageId"] for q in (*train.values(), *held_out.values())}:
            zf.writestr(f"images/{image_id}.jpg", jpeg_bytes())

    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    config["data"]["targets"] = {"train_pool": 8, "internal_val": 2, "test": 3}
    config["data"]["internal_val_image_fraction"] = 0.2
    config["train"]["examples_per_task"] = 4
    config["train"]["effective_batch_size"] = 2
    config["evaluation"]["samples_per_task"] = 3
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return common.load_config(config_path), config_path


def build(cfg):
    assert prepare.cmd_build(cfg, _Args()) == 0
    processed = common.resolve(cfg, "processed_dir")
    return common.read_jsonl(processed / "metadata.jsonl"), json.loads((processed / "manifest.json").read_text())


class _Args:
    skip_extract = False


# 1. Task classification follows structural label and real operation steps.
@pytest.mark.parametrize("task", list(RULES))
def test_classification_rules(workspace, task):
    cfg, _ = workspace
    assert common.matching_tasks(make_question(task, "n0", 1), cfg) == [task]


def test_classification_uses_program_steps_not_semantic_type(workspace):
    cfg, _ = workspace
    question = make_question("T1", "n0", 1)
    question["types"]["semantic"] = "rel"  # the label says relation, the program has no relate step
    assert common.matching_tasks(question, cfg) == ["T1"]


def test_excluded_structural_type_matches_no_group(workspace):
    cfg, _ = workspace
    question = make_question("T1", "n0", 1)
    question["types"]["structural"] = "logical"
    assert common.matching_tasks(question, cfg) == []


def test_detailed_type_crossed_with_operation(workspace):
    cfg, _ = workspace
    cfg["tasks"] = {
        "T1": {"structural": "query", "detailed": "directOf", "has_operation": False},
        "T2": {"structural": "query", "detailed": "directOf", "has_operation": True},
        "T3": {"structural": "query", "detailed": "directWhich", "has_operation": False},
        "T4": {"structural": "query", "detailed": "directWhich", "has_operation": True},
    }
    question = make_question("T1", "n0", 1)
    for detailed, relate, expected in [("directOf", False, "T1"), ("directOf", True, "T2"),
                                       ("directWhich", False, "T3"), ("directWhich", True, "T4")]:
        question["types"]["detailed"] = detailed
        question["semantic"] = [{"operation": "select"}, {"operation": "query"}]
        if relate:
            question["semantic"].insert(1, {"operation": "relate"})
        assert common.matching_tasks(question, cfg) == [expected]


# 2. Every selected sample belongs to exactly one group.
def test_selected_samples_belong_to_one_group(workspace):
    cfg, _ = workspace
    samples, _ = build(cfg)
    for sample in samples:
        synthetic = {"types": {"structural": sample["structural"]},
                     "semantic": [{"operation": op} for op in sample["operations"]]}
        assert common.matching_tasks(synthetic, cfg) == [sample["task"]]


# 3 and 4. No image overlap, no duplicate ids, and violations are detected.
def test_no_leakage_or_duplicates(workspace):
    cfg, _ = workspace
    samples, _ = build(cfg)
    images = {s: {x["image_id"] for x in samples if x["split"] == s} for s in prepare.SPLITS}
    assert not images["train_pool"] & images["internal_val"]
    assert not (images["train_pool"] | images["internal_val"]) & images["test"]
    ids = [s["question_id"] for s in samples]
    assert len(ids) == len(set(ids))


def _valid_rows():
    rows = []
    for split, prefix in (("train_pool", "a"), ("internal_val", "b"), ("test", "c")):
        rows += [{"question_id": f"{split}{t}", "image_id": f"{prefix}{t}", "split": split, "task": t} for t in RULES]
    return rows


def test_validator_rejects_image_leak():
    rows = _valid_rows()
    prepare.validate_selection(rows)
    rows[-1]["image_id"] = rows[0]["image_id"]
    with pytest.raises(ValueError, match="leakage"):
        prepare.validate_selection(rows)


def test_validator_rejects_duplicate_question():
    rows = _valid_rows()
    rows[-1]["question_id"] = rows[0]["question_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        prepare.validate_selection(rows)


def test_selection_is_deterministic(workspace):
    cfg, _ = workspace
    first, _ = build(cfg)
    second, _ = build(cfg)
    assert [s["question_id"] for s in first] == [s["question_id"] for s in second]


def test_insufficient_data_reports_and_does_not_shrink(workspace):
    cfg, _ = workspace
    cfg["data"]["targets"]["test"] = 10_000
    records = prepare.load_eligible(cfg, cfg["data"]["train_questions_file"])
    held_out = prepare.load_eligible(cfg, cfg["data"]["heldout_questions_file"])
    with pytest.raises(prepare.InsufficientData, match="Largest balanced option"):
        prepare.select_samples(records, held_out, cfg)


def test_audit_flags_empty_group_without_changing_definitions(workspace):
    cfg, _ = workspace
    cfg["tasks"]["T1"]["structural"] = "nonexistent"
    assert prepare.audit_split(cfg, cfg["data"]["train_questions_file"])["groups"].get("T1") is None
    audit = {"splits": {"train": prepare.audit_split(cfg, cfg["data"]["train_questions_file"]),
                        "heldout": prepare.audit_split(cfg, cfg["data"]["heldout_questions_file"])}}
    assert any("T1 is EMPTY" in w for w in prepare.audit_warnings(audit, cfg))
    assert cfg["tasks"]["T1"]["structural"] == "nonexistent"


def test_audit_reports_format_confound_and_reads_zip(workspace):
    cfg, _ = workspace
    groups = prepare.audit_split(cfg, cfg["data"]["train_questions_file"])["groups"]
    assert groups["T3"]["yes_no_share"] == 1.0
    assert groups["T1"]["yes_no_share"] == 0.0
    assert groups["T2"]["operation_question_share"]["relate"] == 1.0
    assert groups["unassigned"]["structural"] == {"choose": 30}


# 5. All selected image paths exist, and only referenced images are extracted.
def test_only_referenced_images_are_extracted_and_exist(workspace):
    cfg, _ = workspace
    samples, _ = build(cfg)
    for sample in samples:
        assert Path(sample["image_path"]).is_file()
    extracted = {p.stem for p in common.resolve(cfg, "images_dir").glob("*.jpg")}
    assert extracted == {s["image_id"] for s in samples}


def test_corrupt_and_missing_images_are_reported(workspace):
    cfg, _ = workspace
    with zipfile.ZipFile(common.resolve(cfg, "images_zip"), "a") as zf:
        zf.writestr("images/broken.jpg", b"not a jpeg")
    result = prepare.extract_images(cfg, ["broken", "absent", "n0"])
    assert result["corrupt"] == ["broken"]
    assert result["missing"] == ["absent"]
    assert result["extracted"] == 1


# 6. Generated training records have valid multimodal format.
def test_training_jsonl_format(workspace):
    cfg, _ = workspace
    _, manifest = build(cfg)
    for entry in manifest["experiments"].values():
        for path in (entry["train_file"], entry["val_file"]):
            for record in common.read_jsonl(path):
                assert set(record) == {"messages", "images"}
                user, assistant = record["messages"]
                assert (user["role"], assistant["role"]) == ("user", "assistant")
                assert user["content"].startswith("<image>Answer the question using a short answer only: ")
                assert user["content"].count("<image>") == len(record["images"]) == 1
                assert Path(record["images"][0]).is_absolute() and Path(record["images"][0]).is_file()
                assert assistant["content"]


def test_prompt_reveals_no_task_label_or_answer(workspace):
    cfg, _ = workspace
    samples, _ = build(cfg)
    seen = []

    def generate(image_path, prompt):
        seen.append(prompt)
        return "x", "stop"

    test = [s for s in samples if s["split"] == "test"]
    evaluate.run_inference(test, generate, cfg, common.resolve(cfg, "runs_dir") / "t" / "p.jsonl")
    for prompt, sample in zip(seen, test):
        assert prompt == common.build_user_content(cfg, sample["question"])
        assert sample["task"] not in prompt and "relate" not in prompt


# 7. E1-E4 use identical per-task budgets and the same examples.
def test_budgets_are_identical_across_experiments(workspace):
    cfg, _ = workspace
    samples, manifest = build(cfg)
    budget = cfg["train"]["examples_per_task"]
    reference = {}
    for experiment in common.TRAINABLE:
        rows = prepare.experiment_samples(samples, experiment, "train_pool", cfg)
        tasks = common.EXPERIMENTS[experiment]["train_tasks"]
        assert len(rows) == budget * len(tasks)
        assert manifest["experiments"][experiment]["max_steps"] == budget * len(tasks) // 2
        for task in tasks:
            ids = {r["question_id"] for r in rows if r["task"] == task}
            assert len(ids) == budget
            assert reference.setdefault(task, ids) == ids


def test_default_config_gives_documented_step_counts():
    cfg = common.load_config()
    train = cfg["train"]
    assert train["effective_batch_size"] // train["micro_batch_size"] == 8
    for tasks, steps in ((1, 50), (2, 100), (4, 200)):
        assert prepare.training_steps(tasks * train["examples_per_task"], train["effective_batch_size"]) == steps


def test_experiment_routing_definitions():
    defs = common.EXPERIMENTS
    assert defs["e3_query"]["eval_tasks"] == ["T1", "T2"] and defs["e3_verify"]["eval_tasks"] == ["T3", "T4"]
    assert defs["e4_no_relate"]["eval_tasks"] == ["T1", "T3"] and defs["e4_relate"]["eval_tasks"] == ["T2", "T4"]
    for i in range(1, 5):
        assert defs[f"e1_t{i}"]["eval_tasks"] == list(common.TASKS)


def test_swift_args_only_differ_in_dataset_output_and_steps(workspace):
    cfg, _ = workspace
    _, manifest = build(cfg)
    as_dict = lambda args: dict(zip(args[::2], args[1::2]))
    a = as_dict(prepare.swift_args(cfg, "e1_t1", manifest["experiments"]["e1_t1"]))
    b = as_dict(prepare.swift_args(cfg, "e2_joint", manifest["experiments"]["e2_joint"]))
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"--dataset", "--val_dataset", "--output_dir", "--max_steps", "--save_steps", "--eval_steps"}
    assert a["--quant_bits"] == "4" and a["--freeze_vit"] == "true" and a["--freeze_aligner"] == "true"
    assert a["--gradient_accumulation_steps"] == "2" and a["--per_device_train_batch_size"] == "1"


def test_smoke_args_are_short_and_separate(workspace):
    cfg, _ = workspace
    _, manifest = build(cfg)
    args = prepare.swift_args(cfg, "e1_t1", manifest["experiments"]["e1_t1"], smoke=True)
    options = dict(zip(args[::2], args[1::2]))
    assert int(options["--max_steps"]) <= 5
    assert "/smoke/" in options["--output_dir"] and "--val_dataset" not in options


def test_stale_dataset_is_refused(workspace):
    cfg, _ = workspace
    build(cfg)
    cfg["train"]["examples_per_task"] = 2
    with pytest.raises(SystemExit, match="changed since"):
        prepare.load_manifest(cfg)


# 8. Normalization.
@pytest.mark.parametrize("raw,expected", [
    ("  Red. ", "red"), ("The car", "car"), ("a", "a"), ("Yes!", "yes"), ("T-Shirt", "t shirt"), ("t shirt", "t shirt"),
    ("3.5", "3.5"), ("1,000", "1000"), ("-2", "-2"), ("two", "two"), ("2", "2"), ("man's", "mans"), ("", ""), (None, ""),
    ("<think>\n\n</think>\n\nblue", "blue"), ("<think>unfinished reasoning", ""),
])
def test_normalization(raw, expected):
    assert common.normalize_answer(raw) == expected
    assert common.normalize_answer(common.normalize_answer(raw)) == expected


def test_scoring_marks_invalid_and_truncated_and_stays_exact():
    sample = {"question_id": "1", "image_id": "n", "task": "T1", "question": "q?", "answer": "sofa"}
    assert common.score_prediction(sample, "The sofa.")["correct"]
    assert not common.score_prediction(sample, "couch")["correct"]
    assert common.score_prediction(sample, "   ")["invalid"]
    assert common.score_prediction(sample, "long", "length")["truncated"]


# 9. Metrics and transfer calculations, checked by hand.
def _records(task, correct, wrong, prefix="q"):
    rows = []
    for i in range(correct + wrong):
        rows.append({"question_id": f"{prefix}{task}{i}", "task": task, "correct": i < correct, "invalid": False,
                     "truncated": False, "prediction_normalized": "yes" if i < correct else "x"})
    return rows


def test_macro_gives_tasks_equal_weight_and_differs_from_pooled():
    records = _records("T1", 10, 0) + _records("T2", 0, 10) + _records("T3", 90, 10) + _records("T4", 45, 55)
    metrics = common.compute_metrics(records)
    assert metrics["per_task"]["T3"]["accuracy"] == 0.9
    assert metrics["macro_accuracy"] == pytest.approx((1 + 0 + 0.9 + 0.45) / 4)
    assert metrics["pooled_accuracy"] == pytest.approx(145 / 220)
    assert metrics["query_macro_accuracy"] == pytest.approx(0.5)


def test_macro_undefined_when_tasks_missing():
    assert common.compute_metrics(_records("T1", 1, 1))["macro_accuracy"] is None


def test_directed_transfer_and_interference():
    baseline = {t: 0.5 for t in common.TASKS}
    single = {"T1": {"T1": 0.8, "T2": 0.4, "T3": 0.5, "T4": 0.6}}
    matrix = report.directed_transfer(baseline, single)
    assert matrix["T1"]["T1"] == pytest.approx(0.3) and matrix["T1"]["T2"] == pytest.approx(-0.1)
    assert matrix["T2"]["T1"] is None
    interference = report.joint_interference({"T1": 0.7, "T2": 0.9}, single)
    assert interference["T1"] == pytest.approx(-0.1) and interference["T2"] is None


def test_paired_ci_and_decision_rule():
    a = {str(i): {"correct": True} for i in range(100)}
    b = {str(i): {"correct": i < 50} for i in range(100)}
    result = report.paired_ci(a, b, seed=42)
    assert result["diff"] == pytest.approx(0.5) and result["ci95"][0] > 0
    assert report.paired_ci(a, {"other": {"correct": True}}, seed=42) is None
    decide = lambda low, high: report.research_decision({"diff": (low + high) / 2, "ci95": [low, high]}, 0.0)[0]
    assert decide(0.01, 0.1) == "Promising" and decide(-0.1, -0.01) == "Not supported"
    assert decide(-0.05, 0.05) == "Inconclusive"
    assert report.research_decision(None, None)[0] == "Pending"
    assert report.research_decision({"diff": 0.1, "ci95": [0.05, 0.2]}, 0.5)[0] == "Inconclusive"


# 10. Reports handle incomplete experiments; evaluation runs with a fake model.
def fake_factory(answers_correct):
    def factory(cfg, adapter):
        def generate(image_path, prompt):
            return ("yes" if answers_correct else "wrong answer"), "stop"
        return generate, lambda: None
    return factory


def run_eval(cfg, experiment, **kwargs):
    metadata = common.resolve(cfg, "processed_dir") / "metadata.jsonl"
    return evaluate.evaluate_experiment(experiment, cfg, metadata, generator_factory=fake_factory(True), **kwargs)


def fake_adapter(cfg, experiment, steps):
    checkpoint = common.resolve(cfg, "runs_dir") / experiment / f"checkpoint-{steps}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}")


def test_report_template_without_any_results(workspace):
    cfg, _ = workspace
    summary = report.build_report(cfg)
    reports = common.resolve(cfg, "reports_dir")
    assert summary["completed"] == [] and summary["decision"] == "Pending"
    for name in ("results.csv", "transfer_matrix.csv", "transfer_heatmap.png", "report.md"):
        assert (reports / name).exists()
    text = (reports / "report.md").read_text()
    assert "Template" in text and "not run" in text and "not a result on the official GQA test" in text


def test_report_with_partial_experiments_and_routing(workspace):
    cfg, _ = workspace
    _, manifest = build(cfg)
    assert run_eval(cfg, "e0") == "evaluated"
    assert run_eval(cfg, "e1_t1") == "skipped-missing"
    fake_adapter(cfg, "e1_t1", manifest["experiments"]["e1_t1"]["max_steps"])
    assert run_eval(cfg, "e1_t1") == "evaluated"
    fake_adapter(cfg, "e3_query", manifest["experiments"]["e3_query"]["max_steps"])
    assert run_eval(cfg, "e3_query") == "evaluated"

    predictions = common.read_jsonl(common.resolve(cfg, "runs_dir") / "e3_query" / "predictions.jsonl")
    assert {p["task"] for p in predictions} == {"T1", "T2"}

    summary = report.build_report(cfg)
    assert summary["completed"] == ["e0", "e1_t1", "e3_query"] and summary["decision"] == "Pending"
    assert summary["transfer"]["T1"]["T3"] == pytest.approx(0.0)
    assert summary["transfer"]["T2"]["T1"] is None
    rows = (common.resolve(cfg, "reports_dir") / "results.csv").read_text()
    assert "E3 question-type" in rows and "partial" in rows and "not run" in rows


def test_full_experiment_set_produces_a_decision(workspace):
    cfg, _ = workspace
    _, manifest = build(cfg)
    for experiment in common.TRAINABLE:
        fake_adapter(cfg, experiment, manifest["experiments"][experiment]["max_steps"])
    for experiment in common.EXPERIMENTS:
        assert run_eval(cfg, experiment) == "evaluated"
    summary = report.build_report(cfg)
    assert summary["decision"] == "Inconclusive"  # identical fake predictions mean a zero difference
    assert summary["e4_minus_e3_macro"]["diff"] == 0


def test_smoke_results_are_kept_out_of_the_report(workspace):
    cfg, _ = workspace
    build(cfg)
    assert run_eval(cfg, "e0", limit=1) == "evaluated"
    assert (common.resolve(cfg, "runs_dir") / "smoke" / "e0" / "metrics.json").exists()
    assert report.build_report(cfg)["completed"] == []


def test_truncated_evaluation_is_ignored_with_warning(workspace):
    cfg, _ = workspace
    build(cfg)
    run_eval(cfg, "e0")
    path = common.resolve(cfg, "runs_dir") / "e0" / "predictions.jsonl"
    path.write_text("".join(path.read_text().splitlines(True)[:-1]))
    summary = report.build_report(cfg)
    assert summary["completed"] == [] and any("incomplete" in w for w in summary["warnings"])


def test_evaluation_resumes_without_regenerating(workspace):
    cfg, _ = workspace
    samples, _ = build(cfg)
    test = [s for s in samples if s["split"] == "test"]
    path = common.resolve(cfg, "runs_dir") / "resume" / "predictions.jsonl"
    calls = []

    def generate(image_path, prompt):
        calls.append(prompt)
        return "yes", "stop"

    evaluate.run_inference(test[:2], generate, cfg, path)
    evaluate.run_inference(test, generate, cfg, path)
    assert len(calls) == len(test)
    assert len(common.read_jsonl(path)) == len(test)


def test_early_stopped_adapter_is_rejected(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-10" / "adapter_config.json").write_text("{}")
    assert evaluate.find_adapter(tmp_path, required_steps=50) is None
    assert evaluate.find_adapter(tmp_path, required_steps=10) == tmp_path / "checkpoint-10"
    assert evaluate.find_adapter(tmp_path) == tmp_path / "checkpoint-10"


def test_budget_estimate_arithmetic():
    cfg = common.load_config()
    rows = {r["stage"]: r for r in report.estimate_budget(cfg, sec_per_step=10, sec_per_example=1, load_seconds=0)}
    assert rows["E0"]["optimizer_steps"] == 0 and rows["E0"]["test_examples"] == 600
    assert rows["E1"]["optimizer_steps"] == 200 and rows["E1"]["test_examples"] == 2400
    assert rows["E2"]["optimizer_steps"] == 200 and rows["E3"]["test_examples"] == 600
    assert rows["E1"]["hours"] == pytest.approx((2000 + 2400 + 400) / 3600)


def test_estimate_refuses_to_invent_timings(workspace):
    _, config_path = workspace
    result = subprocess.run([sys.executable, str(ROOT / "src/report.py"), "--config", str(config_path), "--estimate"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "No measured timings" in result.stderr


def test_train_script_resolves_a_command_without_running_it(workspace):
    cfg, config_path = workspace
    build(cfg)
    output = subprocess.run(
        [sys.executable, str(ROOT / "src/prepare.py"), "--config", str(config_path), "train-args", "e1_t1", "--dry-run"],
        capture_output=True, text=True, check=True).stdout
    env, argv = output.split("\n--\n")
    assert "CUDA_VISIBLE_DEVICES=0" in env and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in env
    assert "--tuner_type" in argv and "e1_t1.jsonl" in argv


# 11. Versioned datasets (v2): fresh test set, training seeds, previous adapters, clustered CIs.
def make_v2(tmp_path, cfg_v1, config_path):
    """A second version next to v1: own paths, larger test, three training seeds, v1 excluded from its test."""
    config = yaml.safe_load(config_path.read_text())
    config.update(version="v2_test", train_seeds=[42, 43, 44], per_seed_runs=True, gpu=1,
                  previous_versions={"v1": {"processed_dir": "data/processed", "runs_dir": "runs"}})
    config["paths"].update(processed_dir="data/v2", images_dir="data/v2/images", runs_dir="runs/v2",
                           reports_dir="reports/v2")
    config["data"]["targets"]["test"] = 4
    config["evaluation"]["samples_per_task"] = 4
    path = tmp_path / "config.v2.yaml"
    path.write_text(yaml.safe_dump(config))
    return common.load_config(path), path


@pytest.fixture
def two_versions(tmp_path):
    cfg_v1, path_v1 = make_workspace(tmp_path, held_out_images=40)
    v1_samples, v1_manifest = build(cfg_v1)
    cfg_v2, path_v2 = make_v2(tmp_path, cfg_v1, path_v1)
    v2_samples, v2_manifest = build(cfg_v2)
    return {"cfg_v1": cfg_v1, "v1": v1_samples, "v1_manifest": v1_manifest,
            "cfg_v2": cfg_v2, "path_v2": path_v2, "v2": v2_samples, "v2_manifest": v2_manifest}


def test_new_test_excludes_every_previous_question_and_image(two_versions):
    v1, v2 = two_versions["v1"], two_versions["v2"]
    v2_test = [s for s in v2 if s["split"] == "test"]
    assert {s["question_id"] for s in v2_test}.isdisjoint({s["question_id"] for s in v1})
    assert {s["image_id"] for s in v2_test}.isdisjoint({s["image_id"] for s in v1})
    assert all(sum(s["task"] == t for s in v2_test) == 4 for t in common.TASKS)
    excluded = two_versions["v2_manifest"]["excluded_from_test"]["v1"]
    assert excluded["previous_test_questions"] == 12 and excluded["questions"] == len(v1)
    assert two_versions["v2_manifest"]["version"] == "v2_test"


def test_previous_version_data_is_untouched_and_not_stale(two_versions):
    cfg_v1 = two_versions["cfg_v1"]
    processed = common.resolve(cfg_v1, "processed_dir")
    assert common.read_jsonl(processed / "metadata.jsonl") == two_versions["v1"]
    assert prepare.load_manifest(cfg_v1)["fingerprint"] == two_versions["v1_manifest"]["fingerprint"]
    assert common.resolve(two_versions["cfg_v2"], "processed_dir") != processed


def test_missing_previous_version_refuses_to_build(workspace, tmp_path):
    cfg, _ = workspace
    cfg["previous_versions"] = {"gone": {"processed_dir": "data/nowhere", "runs_dir": "runs"}}
    with pytest.raises(SystemExit, match="fresh test set"):
        prepare.previous_selection(cfg)


def test_validator_rejects_reused_previous_question():
    rows = _valid_rows()
    prepare.validate_selection(rows, {"x"}, {"y"})
    with pytest.raises(ValueError, match="previous version"):
        prepare.validate_selection(rows, {rows[-1]["question_id"]}, set())
    with pytest.raises(ValueError, match="previous version"):
        prepare.validate_selection(rows, set(), {rows[-1]["image_id"]})


def test_training_seed_changes_only_seed_and_output_dir(two_versions):
    cfg, entry = two_versions["cfg_v2"], two_versions["v2_manifest"]["experiments"]["e3_query"]
    as_dict = lambda args: dict(zip(args[::2], args[1::2]))
    base, other = as_dict(prepare.swift_args(cfg, "e3_query", entry)), as_dict(prepare.swift_args(cfg, "e3_query", entry,
                                                                                                    seed=43))
    assert {k for k in base if base[k] != other[k]} == {"--seed", "--data_seed", "--output_dir"}
    assert base["--output_dir"].endswith("runs/v2/seed42/e3_query") and other["--seed"] == "43"
    assert other["--output_dir"].endswith("runs/v2/seed43/e3_query")
    assert common.run_dir(cfg, "e0", seed=43) == common.resolve(cfg, "runs_dir") / "e0"


def test_v1_run_layout_is_unchanged(workspace):
    cfg, _ = workspace
    runs = common.resolve(cfg, "runs_dir")
    assert common.run_dir(cfg, "e3_query") == runs / "e3_query"
    assert common.run_dir(cfg, "e3_query", seed=43) == runs / "seed43" / "e3_query"
    assert common.run_dir(cfg, "e3_query", smoke=True) == runs / "smoke" / "e3_query"


def test_v2_config_budgets_and_isolation():
    cfg = common.load_config(ROOT / "config.v2_1000train_500test.yaml")
    v1 = common.load_config()
    for key in ("model", "model_path", "prompt_template", "tasks", "grouping_operation", "seed"):
        assert cfg[key] == v1[key]
    differing = {k for k in v1["train"] if v1["train"][k] != cfg["train"][k]}
    assert differing == {"examples_per_task"}
    assert cfg["data"]["targets"] == {"train_pool": 1000, "internal_val": 200, "test": 500}
    assert cfg["evaluation"]["samples_per_task"] == 500 and cfg["train_seeds"] == [42, 43, 44] and cfg["gpu"] == 1
    for tasks, steps in ((1, 125), (2, 250), (4, 500)):
        assert prepare.training_steps(tasks * cfg["train"]["examples_per_task"], 8) == steps
    for key in ("processed_dir", "images_dir", "runs_dir", "reports_dir"):
        assert common.resolve(cfg, key) != common.resolve(v1, key)
    assert common.resolve_path(cfg, cfg["previous_versions"]["v1_400train"]["processed_dir"]) == \
        common.resolve(v1, "processed_dir")


def fake_adapter_at(directory, steps):
    (directory / f"checkpoint-{steps}").mkdir(parents=True)
    (directory / f"checkpoint-{steps}" / "adapter_config.json").write_text("{}")


def eval_v2(cfg, experiment, correct=True, **kwargs):
    metadata = common.resolve(cfg, "processed_dir") / "metadata.jsonl"
    return evaluate.evaluate_experiment(experiment, cfg, metadata, generator_factory=fake_factory(correct), **kwargs)


def test_previous_adapters_are_scored_on_the_new_test_without_retraining(two_versions):
    cfg_v1, cfg = two_versions["cfg_v1"], two_versions["cfg_v2"]
    v1_steps = two_versions["v1_manifest"]["experiments"]["e3_query"]["max_steps"]
    assert eval_v2(cfg, "e3_query", adapters_from="v1") == "skipped-missing"
    fake_adapter_at(common.run_dir(cfg_v1, "e3_query"), v1_steps)
    assert eval_v2(cfg, "e3_query", adapters_from="v1") == "evaluated"
    assert eval_v2(cfg, "e0", adapters_from="v1") == "skipped-missing"
    out = common.resolve(cfg, "runs_dir") / "v1" / "e3_query"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["adapter"] == str(common.run_dir(cfg_v1, "e3_query") / f"checkpoint-{v1_steps}")
    assert metrics["adapter_source"] == "v1" and metrics["train_seed"] is None
    v2_test = {s["question_id"] for s in two_versions["v2"] if s["split"] == "test"}
    assert {r["question_id"] for r in common.read_jsonl(out / "predictions.jsonl")} <= v2_test
    assert not (common.run_dir(cfg_v1, "e3_query") / "predictions.jsonl").exists()
    with pytest.raises(SystemExit, match="previous_versions"):
        eval_v2(cfg, "e3_query", adapters_from="unknown")


def test_cluster_bootstrap_matches_question_level_and_widens_with_shared_images():
    a = {"T1": {str(i): float(i < 60) for i in range(100)}}
    b = {"T1": {str(i): float(i < 40) for i in range(100)}}
    one_per_image = report.cluster_paired_ci(a, b, {str(i): f"img{i}" for i in range(100)}, ["T1"], seed=42)
    questions = report.paired_ci({q: {"correct": v} for q, v in a["T1"].items()},
                                 {q: {"correct": v} for q, v in b["T1"].items()}, seed=42)
    assert one_per_image["diff"] == pytest.approx(0.2) == questions["diff"]
    width = lambda r: r["ci95"][1] - r["ci95"][0]
    assert width(one_per_image) == pytest.approx(width(questions), abs=0.03)
    clustered = report.cluster_paired_ci(a, b, {str(i): f"img{i // 10}" for i in range(100)}, ["T1"], seed=42)
    assert clustered["clusters"] == 10 and width(clustered) > width(one_per_image)
    assert report.cluster_paired_ci(a, {"T1": {"x": 1.0}}, {}, ["T1"], seed=42) is None


def test_seed_report_pools_seeds_and_waits_for_all_of_them(two_versions):
    cfg, manifest = two_versions["cfg_v2"], two_versions["v2_manifest"]
    assert eval_v2(cfg, "e0") == "evaluated"
    for seed in (42, 43):
        for experiment in ("e2_joint", "e3_query", "e3_verify", "e4_no_relate", "e4_relate"):
            fake_adapter_at(common.run_dir(cfg, experiment, seed), manifest["experiments"][experiment]["max_steps"])
            assert eval_v2(cfg, experiment, correct=experiment.startswith("e4"), seed=seed) == "evaluated"
    summary = report.build_seed_report(cfg)
    assert summary["decision"] == "Pending" and "2 of 3" in summary["rationale"]
    # fake model: E3 adapters always wrong, E4 adapters answer "yes" (right on verify questions whose answer is yes)
    acc = summary["accuracy"]
    expected = acc["seed42 | E4 operation grouping"]["macro_accuracy"] - acc["seed42 | E3 question-type"]["macro_accuracy"]
    assert expected > 0
    assert summary["e4_minus_e3_macro_per_seed"] == {"seed42": pytest.approx(expected), "seed43": pytest.approx(expected)}

    for experiment in ("e2_joint", "e3_query", "e3_verify", "e4_no_relate", "e4_relate"):
        fake_adapter_at(common.run_dir(cfg, experiment, 44), manifest["experiments"][experiment]["max_steps"])
        eval_v2(cfg, experiment, correct=experiment.startswith("e4"), seed=44)
    summary = report.build_seed_report(cfg)
    assert summary["transfer_mean_over_seeds"] is None and summary["interference_e2_minus_e1"] is None
    for seed in (42, 43, 44):  # E0 and E1 answer "yes"; E2 answers "wrong answer"
        for experiment in common.EXPERIMENT_GROUPS["e1"]:
            fake_adapter_at(common.run_dir(cfg, experiment, seed), manifest["experiments"][experiment]["max_steps"])
            assert eval_v2(cfg, experiment, correct=True, seed=seed) == "evaluated"
    summary = report.build_seed_report(cfg)
    acc = summary["accuracy"]
    e1_t3 = acc["seed42 | E1 single-task"]["per_task"]["T3"]["accuracy"]
    e0_t3 = acc["pretrained | E0 pretrained"]["per_task"]["T3"]["accuracy"]
    assert e1_t3 > 0 and summary["transfer_seeds"] == 3
    assert summary["transfer_mean_over_seeds"]["T1"]["T3"] == pytest.approx(e1_t3 - e0_t3)
    assert summary["interference_seeds"] == 3
    assert summary["interference_e2_minus_e1"]["T3"]["diff"] == pytest.approx(-e1_t3)
    pooled = summary["e4_minus_e3"]["seeds pooled (3/3)"]["Macro (T1-T4)"]
    assert pooled["diff"] == pytest.approx(expected) and pooled["ci95"][0] <= expected <= pooled["ci95"][1]
    assert summary["decision"] in ("Promising", "Inconclusive")  # a tiny synthetic test gives a wide CI
    reports = common.resolve(cfg, "reports_dir")
    text = (reports / "report.md").read_text()
    assert "seed44" in text and "mean ± sd (3 seeds)" in text and "bootstrap over images" in text
    assert "## Directed transfer (E1 vs E0)" in text and "## MTL interference (E2 − E1)" in text
    assert "v2_test" in text and (reports / "results.csv").exists()
    assert not (common.resolve(two_versions["cfg_v1"], "reports_dir") / "report.md").exists()


def test_train_script_dry_run_accepts_config_and_seed(two_versions):
    env = {**__import__("os").environ, "PYTHON": sys.executable}
    output = subprocess.run(["bash", str(ROOT / "scripts/train.sh"), "e4_relate", "--config", str(two_versions["path_v2"]),
                             "--seed", "44", "--dry-run"], capture_output=True, text=True, env=env, check=True).stdout
    assert "CUDA_VISIBLE_DEVICES=1" in output and "runs/v2/seed44/e4_relate" in output
    assert "--seed \\\n  44" in output and "data/v2/train/e4_relate.jsonl" in output
