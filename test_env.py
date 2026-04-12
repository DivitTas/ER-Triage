import pytest

from ER_Triage import ErTriageAction, TriagePriority
from ER_Triage.server import ErTriageEnvironment


TASK_PATIENT_COUNTS = {
    "task_1": 5,
    "task_2": 10,
    "task_3": 15,
}


def test_reset_returns_first_patient_state() -> None:
    env = ErTriageEnvironment(task_id="task_1")

    obs = env.reset()

    assert obs.patient_id == 1
    assert obs.patients_remaining == 4
    assert obs.critical_beds_available == 2
    assert obs.current_patient_wait_time == 0
    assert obs.done is False
    assert 70 <= obs.oxygen_saturation <= 100


@pytest.mark.parametrize("task_id,expected_steps", TASK_PATIENT_COUNTS.items())
def test_each_task_terminates_with_expected_patient_count(
    task_id: str, expected_steps: int
) -> None:
    env = ErTriageEnvironment(task_id=task_id)
    obs = env.reset(task_id=task_id)

    rewards = []
    steps = 0

    while not obs.done:
        obs = env.step(ErTriageAction(priority=TriagePriority.URGENT))
        rewards.append(obs.reward)
        steps += 1

        assert 0.0 <= obs.reward <= 1.0
        assert 0 <= obs.critical_beds_available <= 2

    assert steps == expected_steps
    assert len(rewards) == expected_steps
    assert obs.done is True
    assert obs.patients_remaining == 0


def test_assigning_critical_priority_uses_a_bed() -> None:
    env = ErTriageEnvironment(task_id="task_3")
    env.reset()

    obs = env.step(ErTriageAction(priority=TriagePriority.CRITICAL))

    assert obs.critical_beds_available in {0, 1}
    assert obs.current_patient_wait_time >= 1


def test_state_step_count_tracks_actions() -> None:
    env = ErTriageEnvironment(task_id="task_1")
    env.reset()

    env.step(ErTriageAction(priority=TriagePriority.URGENT))
    env.step(ErTriageAction(priority=TriagePriority.URGENT))

    assert env.state.step_count == 2


def test_constructor_rejects_invalid_task_id() -> None:
    with pytest.raises(ValueError, match="Unknown task_id 'task_x'"):
        ErTriageEnvironment(task_id="task_x")


def test_reset_rejects_invalid_task_id_without_mutating_existing_task() -> None:
    env = ErTriageEnvironment(task_id="task_2")

    with pytest.raises(ValueError, match="Unknown task_id 'task_x'"):
        env.reset(task_id="task_x")

    assert env._task_id == "task_2"


def test_one_level_miss_penalizes_high_acuity_more() -> None:
    env = ErTriageEnvironment(task_id="task_1")

    high_acuity_reward = env._calculate_reward(
        ErTriageAction(priority=TriagePriority.EMERGENT),
        {"true_priority": 1, "deteriorated": False},
        bed_was_available=True,
    )
    low_acuity_reward = env._calculate_reward(
        ErTriageAction(priority=TriagePriority.LESS_URGENT),
        {"true_priority": 5, "deteriorated": False},
        bed_was_available=True,
    )

    assert 0.0 <= high_acuity_reward <= 1.0
    assert 0.0 <= low_acuity_reward <= 1.0
    assert high_acuity_reward < low_acuity_reward


def test_deterioration_caps_reward_without_flattening_quality() -> None:
    env = ErTriageEnvironment(task_id="task_1")

    correct_reward = env._calculate_reward(
        ErTriageAction(priority=TriagePriority.EMERGENT),
        {"true_priority": 2, "deteriorated": True},
        bed_was_available=True,
    )
    incorrect_reward = env._calculate_reward(
        ErTriageAction(priority=TriagePriority.NON_URGENT),
        {"true_priority": 2, "deteriorated": True},
        bed_was_available=True,
    )

    assert 0.0 <= incorrect_reward < correct_reward <= 0.3
