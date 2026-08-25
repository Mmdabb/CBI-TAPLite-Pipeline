from corridor_measurement import runtime


def test_worker_plan_targets_half_of_currently_free_cores(monkeypatch):
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 40)
    monkeypatch.setattr(
        runtime,
        "_measure_free_core_equivalents",
        lambda logical_cores, sample_seconds: (30.0, "test_measurement"),
    )

    plan = runtime.recommend_workers(70, target_fraction=0.50)

    assert plan.logical_cores == 40
    assert plan.free_core_equivalents == 30.0
    assert plan.workers == 15
    assert plan.threads_per_worker == 1


def test_worker_plan_caps_at_half_of_all_cores(monkeypatch):
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 40)
    monkeypatch.setattr(
        runtime,
        "_measure_free_core_equivalents",
        lambda logical_cores, sample_seconds: (40.0, "test_measurement"),
    )

    plan = runtime.recommend_workers(70, target_fraction=0.50)

    assert plan.workers == 20
