import math

import pytest

from ER_Triage.grader import grade_episode, grade_task, grader


def test_grade_episode_returns_zero_for_empty_rewards() -> None:
    assert grade_episode([]) == 0.0
    assert grader(None) == 0.0


def test_grade_episode_clamps_invalid_and_out_of_range_values() -> None:
    rewards = [1.5, -0.25, "0.4", None, float("nan"), float("inf")]

    assert grade_episode(rewards) == pytest.approx((1.0 + 0.0 + 0.4 + 0.0 + 0.0 + 0.0) / 6)


def test_grade_task_stays_bounded_for_malformed_episode_inputs() -> None:
    score = grade_task([[2.0, -1.0], [float("nan")], None, "0.75"])

    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx((0.5 + 0.0 + 0.0 + 0.75) / 4)
