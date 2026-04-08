# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the ER Triage Environment.

The ER_Triage environment simulates patient triage based on vital signs.
An agent must assign priority levels (1-5) to patients based on their vitals.
"""

from enum import Enum
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TriagePriority(int, Enum):
    """Triage priority levels (Emergency Severity Index style)."""
    CRITICAL = 1      # Immediate life threat - needs resuscitation
    EMERGENT = 2      # High risk - needs rapid intervention
    URGENT = 3        # Stable but needs timely care
    LESS_URGENT = 4   # Can wait 1-2 hours
    NON_URGENT = 5    # Minor issue - can wait


class ErTriageAction(Action):
    """Action for ER Triage - assign priority level to current patient."""

    priority: TriagePriority = Field(..., description="Triage priority 1-5 (1=critical, 5=non-urgent)")


class ErTriageObservation(Observation):
    """Observation showing current patient's vital signs and resource state."""

    patient_id: int = Field(..., description="Unique patient identifier")
    systolic_bp: int = Field(..., description="Systolic blood pressure (mmHg)")
    diastolic_bp: int = Field(..., description="Diastolic blood pressure (mmHg)")
    heart_rate: int = Field(..., description="Heart rate (bpm)")
    respiratory_rate: int = Field(..., description="Respiratory rate (breaths/min)")
    temperature: float = Field(..., description="Body temperature (Celsius)")
    oxygen_saturation: int = Field(..., description="SpO2 percentage")
    chief_complaint: str = Field(..., description="Patient's main complaint")
    patients_remaining: int = Field(..., description="Patients left in queue")
    critical_beds_available: int = Field(..., description="Available critical care beds (0-2)")
    current_patient_wait_time: int = Field(..., description="Steps this patient has waited")

