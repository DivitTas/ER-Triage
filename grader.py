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

from typing import List


def grade_episode(rewards: List[float]) -> float:
    """
    Grade an episode based on per-step rewards.
    
    Args:
        rewards: List of rewards from each step (already in 0.0-1.0 range)
    
    Returns:
        Final score in 0.0-1.0 range (average of rewards)
    """
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def grade_task(task_rewards: List[List[float]]) -> float:
    """
    Grade multiple episodes of the same task.
    
    Args:
        task_rewards: List of reward lists, one per episode
    
    Returns:
        Average score across all episodes (0.0-1.0)
    """
    if not task_rewards:
        return 0.0
    episode_scores = [grade_episode(r) for r in task_rewards]
    return sum(episode_scores) / len(episode_scores)


# For hackathon validator compatibility
def grader(rewards: List[float]) -> float:
    """Alias for grade_episode - hackathon validator entry point."""
    return grade_episode(rewards)
