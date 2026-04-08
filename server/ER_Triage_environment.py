# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ER Triage Environment Implementation.

Simulates patient triage based on vital signs. Agent assigns priority 1-5.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ErTriageAction, ErTriageObservation, TriagePriority
except ImportError:
    from models import ErTriageAction, ErTriageObservation, TriagePriority


class ErTriageEnvironment(Environment):
    """
    ER Triage environment where agent prioritizes patients based on vital signs.

    Example:
        >>> env = ErTriageEnvironment()
        >>> obs = env.reset()
        >>> print(f"Patient {obs.patient_id}: HR={obs.heart_rate}, BP={obs.systolic_bp}/{obs.diastolic_bp}")
        >>>
        >>> obs = env.step(ErTriageAction(priority=TriagePriority.URGENT))
        >>> print(f"Reward: {obs.reward}, Remaining: {obs.patients_remaining}")
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the ER_Triage environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._patients = []
        self._current_idx = 0

    def reset(self) -> ErTriageObservation:
        """
        Reset the environment with a new queue of patients.

        Returns:
            ErTriageObservation with first patient's vital signs
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_idx = 0
        
        # TODO: Generate patient queue
        # For now, return placeholder patient
        return ErTriageObservation(
            patient_id=1,
            systolic_bp=120,
            diastolic_bp=80,
            heart_rate=72,
            respiratory_rate=16,
            temperature=37.0,
            oxygen_saturation=98,
            chief_complaint="placeholder - implement patient generator",
            patients_remaining=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: ErTriageAction) -> ErTriageObservation:  # type: ignore[override]
        """
        Process triage decision and return next patient.

        Args:
            action: ErTriageAction with priority assignment

        Returns:
            ErTriageObservation with next patient or done=True
        """
        self._state.step_count += 1

        # TODO: Calculate reward based on action.priority vs ground truth
        reward = 0.5  # placeholder

        # TODO: Return next patient or mark done
        return ErTriageObservation(
            patient_id=1,
            systolic_bp=120,
            diastolic_bp=80,
            heart_rate=72,
            respiratory_rate=16,
            temperature=37.0,
            oxygen_saturation=98,
            chief_complaint="placeholder - implement step logic",
            patients_remaining=0,
            done=True,
            reward=reward,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
