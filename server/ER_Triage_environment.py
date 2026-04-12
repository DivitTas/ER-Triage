# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ER Triage Environment Implementation.

Simulates patient triage with resource constraints (critical beds) and time pressure
(patient deterioration). This creates a true RL problem with sequential dependencies.
"""

import math
import random
from uuid import uuid4
from typing import List, Dict

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ErTriageAction, ErTriageObservation, TriagePriority
except ImportError:
    from models import ErTriageAction, ErTriageObservation, TriagePriority


# Vital sign ranges by priority level
VITAL_RANGES = {
    1: {  # CRITICAL - immediate life threat
        "systolic_bp": (50, 80),      # or (200, 250) for hypertensive crisis
        "diastolic_bp": (30, 50),
        "heart_rate": (150, 200),      # or (20, 40) for bradycardia
        "respiratory_rate": (30, 45),  # or (4, 8) for respiratory failure
        "temperature": (35.0, 35.5),   # hypothermia or (40.5, 42.0) hyperthermia
        "oxygen_saturation": (70, 85),
        "complaints": ["unresponsive", "not breathing", "severe bleeding", "cardiac arrest", "choking"],
    },
    2: {  # EMERGENT - high risk
        "systolic_bp": (80, 95),
        "diastolic_bp": (50, 60),
        "heart_rate": (120, 150),
        "respiratory_rate": (24, 30),
        "temperature": (39.5, 40.5),
        "oxygen_saturation": (85, 90),
        "complaints": ["severe chest pain", "stroke symptoms", "difficulty breathing", "severe abdominal pain", "high fever with confusion"],
    },
    3: {  # URGENT - stable but needs timely care
        "systolic_bp": (95, 110),
        "diastolic_bp": (60, 70),
        "heart_rate": (100, 120),
        "respiratory_rate": (20, 24),
        "temperature": (38.5, 39.5),
        "oxygen_saturation": (90, 94),
        "complaints": ["moderate pain", "persistent vomiting", "minor head injury", "deep cut needing stitches", "asthma attack"],
    },
    4: {  # LESS URGENT - can wait 1-2 hours
        "systolic_bp": (110, 130),
        "diastolic_bp": (70, 85),
        "heart_rate": (80, 100),
        "respiratory_rate": (16, 20),
        "temperature": (37.5, 38.5),
        "oxygen_saturation": (94, 96),
        "complaints": ["mild pain", "sprained ankle", "earache", "urinary symptoms", "mild rash"],
    },
    5: {  # NON-URGENT - minor issue
        "systolic_bp": (110, 130),
        "diastolic_bp": (70, 80),
        "heart_rate": (60, 80),
        "respiratory_rate": (12, 16),
        "temperature": (36.5, 37.5),
        "oxygen_saturation": (96, 100),
        "complaints": ["prescription refill", "minor cold symptoms", "small cut", "insect bite", "medication question"],
    },
}

# Task configurations
TASK_CONFIG = {
    "task_1": {"num_patients": 5, "priority_weights": [0.1, 0.2, 0.3, 0.2, 0.2]},   # Easy
    "task_2": {"num_patients": 10, "priority_weights": [0.2, 0.25, 0.25, 0.2, 0.1]}, # Medium
    "task_3": {"num_patients": 15, "priority_weights": [0.3, 0.3, 0.2, 0.1, 0.1]},   # Hard (more critical)
}
VALID_TASK_IDS = tuple(TASK_CONFIG.keys())

BED_COOLDOWN_STEPS = 4  # Steps until a bed frees up
MAX_WAIT_BEFORE_DETERIORATION = 3  # Steps before P1/P2 patients deteriorate


class ErTriageEnvironment(Environment):
    """
    ER Triage environment with resource constraints and time pressure.
    
    RL Elements:
    - Resource scarcity: 2 critical beds, used when assigning Priority 1
    - Time pressure: Patients deteriorate if they wait too long
    - Sequential dependency: Bed usage affects future availability
    - Trade-offs: Over-triage wastes beds, under-triage harms patients
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "task_1"):
        """Initialize the ER_Triage environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = self._validate_task_id(task_id)
        self._patients: List[Dict] = []
        self._current_idx = 0
        self._critical_beds = 2
        self._bed_free_at: List[int] = []  # Step counts when beds free up
        self._global_step = 0

    @staticmethod
    def _validate_task_id(task_id: str) -> str:
        """Validate task IDs against the supported task configuration keys."""
        if task_id not in TASK_CONFIG:
            valid_task_ids = ", ".join(VALID_TASK_IDS)
            raise ValueError(f"Unknown task_id '{task_id}'. Expected one of: {valid_task_ids}")
        return task_id

    def _generate_patient(self, patient_id: int, priority: int) -> Dict:
        """Generate a patient with vitals matching the given priority."""
        ranges = VITAL_RANGES[priority]
        
        # Randomly decide if some vitals should be inverted (e.g., low BP vs high BP)
        systolic = random.randint(*ranges["systolic_bp"])
        if priority == 1 and random.random() > 0.5:
            systolic = random.randint(200, 250)  # Hypertensive crisis variant
        
        heart_rate = random.randint(*ranges["heart_rate"])
        if priority == 1 and random.random() > 0.7:
            heart_rate = random.randint(20, 40)  # Bradycardia variant
            
        return {
            "patient_id": patient_id,
            "systolic_bp": systolic,
            "diastolic_bp": random.randint(*ranges["diastolic_bp"]),
            "heart_rate": heart_rate,
            "respiratory_rate": random.randint(*ranges["respiratory_rate"]),
            "temperature": round(random.uniform(*ranges["temperature"]), 1),
            "oxygen_saturation": random.randint(*ranges["oxygen_saturation"]),
            "chief_complaint": random.choice(ranges["complaints"]),
            "true_priority": priority,
            "wait_time": 0,
            "deteriorated": False,
        }

    def _calculate_true_priority(self, patient: Dict) -> int:
        """Determine ground truth priority from vitals (for validation/grading)."""
        # This returns the stored true_priority since we generate patients by priority
        # In a more complex version, this would analyze vitals directly
        return patient["true_priority"]

    def _update_beds(self):
        """Free up beds that have completed their cooldown."""
        self._bed_free_at = [t for t in self._bed_free_at if t > self._global_step]
        self._critical_beds = 2 - len(self._bed_free_at)

    def _use_bed(self):
        """Mark a bed as in use for BED_COOLDOWN_STEPS."""
        if self._critical_beds > 0:
            self._bed_free_at.append(self._global_step + BED_COOLDOWN_STEPS)
            self._critical_beds -= 1
            return True
        return False

    def _calculate_reward(self, action: ErTriageAction, patient: Dict, bed_was_available: bool) -> float:
        """Calculate reward based on triage decision, resources, and patient state."""
        true_priority = patient["true_priority"]
        assigned_priority = action.priority.value
        priority_delta = assigned_priority - true_priority
        priority_diff = abs(priority_delta)

        if priority_diff == 0:
            base_reward = 1.0
        else:
            # P1 should be punished much more than P5 for the same miss distance.
            criticality = (6 - true_priority) / 5.0  # P1=1.0 ... P5=0.2
            if priority_delta > 0:
                # Under-triage: assigning too low urgency.
                exponent = priority_diff * (0.6 + criticality)
            else:
                # Over-triage: less dangerous, still penalized.
                exponent = priority_diff * (0.1 + (0.6 * criticality))
            base_reward = math.exp(-exponent)

        if assigned_priority == 1 and not bed_was_available:
            # Preserve existing transfer penalty when no critical bed is available.
            base_reward *= 0.6

        if patient["deteriorated"]:
            # Keep action-quality ordering while capping upside after delay damage.
            base_reward *= 0.3

        return max(0.0, min(base_reward, 1.0))

    def _get_current_observation(self, reward: float = 0.0, done: bool = False) -> ErTriageObservation:
        """Build observation for current patient."""
        if self._current_idx >= len(self._patients):
            # No more patients - return final observation
            last_patient = self._patients[-1] if self._patients else None
            return ErTriageObservation(
                patient_id=last_patient["patient_id"] if last_patient else 0,
                systolic_bp=last_patient["systolic_bp"] if last_patient else 120,
                diastolic_bp=last_patient["diastolic_bp"] if last_patient else 80,
                heart_rate=last_patient["heart_rate"] if last_patient else 72,
                respiratory_rate=last_patient["respiratory_rate"] if last_patient else 16,
                temperature=last_patient["temperature"] if last_patient else 37.0,
                oxygen_saturation=last_patient["oxygen_saturation"] if last_patient else 98,
                chief_complaint="Episode complete",
                patients_remaining=0,
                critical_beds_available=self._critical_beds,
                current_patient_wait_time=0,
                done=True,
                reward=reward,
            )
        
        patient = self._patients[self._current_idx]
        return ErTriageObservation(
            patient_id=patient["patient_id"],
            systolic_bp=patient["systolic_bp"],
            diastolic_bp=patient["diastolic_bp"],
            heart_rate=patient["heart_rate"],
            respiratory_rate=patient["respiratory_rate"],
            temperature=patient["temperature"],
            oxygen_saturation=patient["oxygen_saturation"],
            chief_complaint=patient["chief_complaint"],
            patients_remaining=len(self._patients) - self._current_idx - 1,
            critical_beds_available=self._critical_beds,
            current_patient_wait_time=patient["wait_time"],
            done=done,
            reward=reward,
        )

    def reset(self, task_id: str = None) -> ErTriageObservation:
        """
        Reset the environment with a new queue of patients.
        
        Args:
            task_id: Optional task identifier ("task_1", "task_2", "task_3")
        """
        if task_id is not None:
            self._task_id = self._validate_task_id(task_id)
            
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_idx = 0
        self._critical_beds = 2
        self._bed_free_at = []
        self._global_step = 0
        
        # Generate patients based on task config
        config = TASK_CONFIG[self._task_id]
        num_patients = config["num_patients"]
        weights = config["priority_weights"]
        
        self._patients = []
        for i in range(num_patients):
            priority = random.choices([1, 2, 3, 4, 5], weights=weights)[0]
            self._patients.append(self._generate_patient(i + 1, priority))
        
        return self._get_current_observation(reward=0.0, done=False)

    def step(self, action: ErTriageAction) -> ErTriageObservation:
        """
        Process triage decision for current patient, return next patient.
        
        Args:
            action: ErTriageAction with priority assignment
            
        Returns:
            ErTriageObservation with next patient's vitals or done=True
        """
        self._state.step_count += 1
        self._global_step += 1
        
        # Update bed availability
        self._update_beds()
        
        # Get current patient
        if self._current_idx >= len(self._patients):
            return self._get_current_observation(reward=0.0, done=True)
        
        patient = self._patients[self._current_idx]
        
        # Check for deterioration (waited too long)
        if patient["wait_time"] > MAX_WAIT_BEFORE_DETERIORATION and patient["true_priority"] <= 2:
            patient["deteriorated"] = True
        
        # Handle bed usage for Priority 1 assignments
        bed_was_available = True
        if action.priority == TriagePriority.CRITICAL:
            bed_was_available = self._use_bed()
        
        # Calculate reward
        reward = self._calculate_reward(action, patient, bed_was_available)
        
        # Move to next patient
        self._current_idx += 1
        
        # Increment wait time for remaining patients
        for i in range(self._current_idx, len(self._patients)):
            self._patients[i]["wait_time"] += 1
        
        # Check if episode is done
        done = self._current_idx >= len(self._patients)
        
        return self._get_current_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
