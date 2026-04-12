# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ER Triage Episode Grader.

Evaluates agent performance on triage tasks.
Returns scores in 0.0-1.0 range as required by hackathon.
"""

import math
from collections.abc import Mapping
from typing import Any, List


def _clamp_score(value: Any) -> float:
    """Coerce arbitrary values into a finite score in the 0.0-1.0 range."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(score):
        return 0.0

    return max(0.0, min(score, 1.0))


def _coerce_sequence(values: Any) -> List[Any]:
    """Normalize scalars, iterables, and mappings into a list-like payload."""
    if values is None:
        return []

    if isinstance(values, Mapping):
        values = values.values()

    if isinstance(values, (str, bytes)):
        return [values]

    try:
        return list(values)
    except TypeError:
        return [values]


def grade_episode(rewards: List[float]) -> float:
    """
    Grade an episode based on per-step rewards.
    
    Args:
        rewards: List of rewards from each step (already in 0.0-1.0 range)
    
    Returns:
        Final score in 0.0-1.0 range (average of rewards)
    """
    normalized_rewards = [_clamp_score(reward) for reward in _coerce_sequence(rewards)]
    if not normalized_rewards:
        return 0.0
    return _clamp_score(sum(normalized_rewards) / len(normalized_rewards))


def grade_task(task_rewards: List[List[float]]) -> float:
    """
    Grade multiple episodes of the same task.
    
    Args:
        task_rewards: List of reward lists, one per episode
    
    Returns:
        Average score across all episodes (0.0-1.0)
    """
    normalized_task_rewards = _coerce_sequence(task_rewards)
    if not normalized_task_rewards:
        return 0.0
    episode_scores = [grade_episode(rewards) for rewards in normalized_task_rewards]
    return _clamp_score(sum(episode_scores) / len(episode_scores))


# For hackathon validator compatibility
def grader(rewards: List[float]) -> float:
    """Alias for grade_episode - hackathon validator entry point."""
    return grade_episode(rewards)
