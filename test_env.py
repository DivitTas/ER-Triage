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
