# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Er Triage Environment.

The ER_Triage environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ErTriageAction(Action):
    """Action for the Er Triage environment - just a message to echo."""

    message: str = Field(..., description="Message to echo back")


class ErTriageObservation(Observation):
    """Observation from the Er Triage environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
