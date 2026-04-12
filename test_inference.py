import importlib.util
from pathlib import Path

import pytest


def _load_inference_module():
    module_path = Path(__file__).with_name("inference.py")
    spec = importlib.util.spec_from_file_location("er_triage_inference", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inference = _load_inference_module()


@pytest.mark.parametrize(
    ("response", "expected_priority"),
    [
        ("1", 1),
        ("Priority: 4", 4),
        ("I choose 5 for this patient.", 5),
    ],
)
def test_parse_priority_accepts_responses_with_1_to_5(
    response: str, expected_priority: int
) -> None:
    assert inference.parse_priority(response) == expected_priority


@pytest.mark.parametrize("response", ["", "urgent", "Priority: six"])
def test_parse_priority_rejects_non_numeric_responses(response: str) -> None:
    with pytest.raises(ValueError, match="valid triage priority"):
        inference.parse_priority(response)
