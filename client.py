# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ER Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ErTriageAction, ErTriageObservation, TriagePriority


class ErTriageEnv(
    EnvClient[ErTriageAction, ErTriageObservation, State]
):
    """
    Client for the ER Triage Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with ErTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(f"Patient {result.observation.patient_id}: HR={result.observation.heart_rate}")
        ...
        ...     result = client.step(ErTriageAction(priority=TriagePriority.URGENT))
        ...     print(f"Reward: {result.reward}, Remaining: {result.observation.patients_remaining}")
    """

    def _step_payload(self, action: ErTriageAction) -> Dict:
        """Convert ErTriageAction to JSON payload for step message."""
        return {
            "priority": action.priority.value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ErTriageObservation]:
        """Parse server response into StepResult[ErTriageObservation]."""
        obs_data = payload.get("observation", {})
        observation = ErTriageObservation(
            patient_id=obs_data.get("patient_id", 0),
            systolic_bp=obs_data.get("systolic_bp", 0),
            diastolic_bp=obs_data.get("diastolic_bp", 0),
            heart_rate=obs_data.get("heart_rate", 0),
            respiratory_rate=obs_data.get("respiratory_rate", 0),
            temperature=obs_data.get("temperature", 0.0),
            oxygen_saturation=obs_data.get("oxygen_saturation", 0),
            chief_complaint=obs_data.get("chief_complaint", ""),
            patients_remaining=obs_data.get("patients_remaining", 0),
            critical_beds_available=obs_data.get("critical_beds_available", 0),
            current_patient_wait_time=obs_data.get("current_patient_wait_time", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
