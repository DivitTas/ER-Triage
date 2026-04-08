"""
ER Triage Inference Script
===================================
Runs an LLM agent through the ER Triage environment for all 3 tasks.

Environment Variables (from .env):
    OPENAI_BASE_URL - Optional override for the OpenAI-compatible API endpoint
    OPENAI_MODEL    - Optional override for the default OpenAI model
    OPENAI_API_KEY  - Your OpenAI API key

Legacy Hugging Face compatibility:
    API_BASE_URL    - Legacy Hugging Face router endpoint override
    MODEL_NAME      - Legacy Hugging Face model name
    HF_TOKEN        - Legacy Hugging Face token

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import re
import sys
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from ER_Triage import ErTriageAction, TriagePriority
from ER_Triage.server import ErTriageEnvironment

# Load environment variables from .env
load_dotenv()

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def resolve_llm_config() -> tuple[str, str, str]:
    """Resolve provider settings with OpenAI defaults and legacy HF fallback."""
    openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    openai_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    hf_api_key = os.getenv("HF_TOKEN")
    hf_base_url = os.getenv("API_BASE_URL")
    hf_model = os.getenv("MODEL_NAME", DEFAULT_HF_MODEL)

    # Preserve existing Hugging Face setups when OpenAI credentials are absent.
    if not openai_api_key and (hf_api_key or hf_base_url):
        return hf_base_url or DEFAULT_HF_BASE_URL, hf_model, hf_api_key or ""

    return openai_base_url, openai_model, openai_api_key or ""


# Configuration
API_BASE_URL, MODEL_NAME, API_KEY = resolve_llm_config()

BENCHMARK = "er_triage"
TASKS = ["task_1", "task_2", "task_3"]
MAX_STEPS = 20  # Maximum steps per task (task_3 has 15 patients max)
TEMPERATURE = 0.3  # Lower temp for more consistent medical decisions
MAX_TOKENS = 100

SYSTEM_PROMPT = textwrap.dedent("""
    You are an experienced ER triage nurse. Your job is to assign triage priority levels 
    to patients based on their vital signs.
    
    Priority Levels:
    1 - CRITICAL: Immediate life threat (cardiac arrest, severe trauma, not breathing)
    2 - EMERGENT: High risk, needs rapid care (chest pain, stroke symptoms, severe bleeding)
    3 - URGENT: Stable but needs timely care (high fever, moderate pain, minor injuries)
    4 - LESS URGENT: Can wait 1-2 hours (mild pain, minor cuts, sprains)
    5 - NON-URGENT: Minor issues (prescription refill, cold symptoms, insect bites)
    
    Consider:
    - Blood pressure: Normal 90-140/60-90 mmHg. <90 systolic = shock, >180 = crisis
    - Heart rate: Normal 60-100 bpm. <50 or >120 = concern
    - Respiratory rate: Normal 12-20/min. <10 or >24 = distress
    - Temperature: Normal 36.5-37.5°C. >38.5 = fever, >40 = high fever
    - Oxygen saturation: Normal 95-100%. <90 = critical
    
    You have LIMITED critical care beds. Use Priority 1 wisely.
    
    Respond with ONLY a single number 1-5. Nothing else.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_patient_prompt(obs) -> str:
    """Build a prompt showing current patient vitals."""
    return textwrap.dedent(f"""
        PATIENT #{obs.patient_id}
        
        Vital Signs:
        - Blood Pressure: {obs.systolic_bp}/{obs.diastolic_bp} mmHg
        - Heart Rate: {obs.heart_rate} bpm
        - Respiratory Rate: {obs.respiratory_rate} breaths/min
        - Temperature: {obs.temperature}°C
        - Oxygen Saturation: {obs.oxygen_saturation}%
        
        Chief Complaint: {obs.chief_complaint}
        
        Resources:
        - Critical beds available: {obs.critical_beds_available}/2
        - Patient wait time: {obs.current_patient_wait_time} steps
        - Patients remaining: {obs.patients_remaining}
        
        Assign triage priority (1-5):
    """).strip()


def parse_priority(response: str) -> int:
    """Extract priority number from LLM response."""
    # Look for a number 1-5 in the response
    match = re.search(r'[1-5]', response)
    if match:
        return int(match.group())
    # Default to 3 (urgent) if parsing fails
    return 3


def get_triage_decision(client: OpenAI, obs) -> tuple[int, Optional[str]]:
    """Get triage priority from LLM based on patient vitals."""
    user_prompt = build_patient_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response = (completion.choices[0].message.content or "").strip()
        return parse_priority(response), None
    except Exception as exc:
        sanitized_error = " ".join(str(exc).split())
        return 3, sanitized_error or "model_request_failed"


def run_task(client: OpenAI, task_id: str) -> dict:
    """Run a single task and return results."""
    # Create environment directly (no Docker needed for local testing)
    env = ErTriageEnvironment(task_id=task_id)
    
    rewards: List[float] = []
    steps_taken = 0
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # Reset environment
        obs = env.reset(task_id=task_id)
        
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break
            
            # Get LLM triage decision
            priority_num, error = get_triage_decision(client, obs)
            priority = TriagePriority(priority_num)
            action = ErTriageAction(priority=priority)
            
            # Take step
            obs = env.step(action)
            
            reward = obs.reward
            done = obs.done
            rewards.append(reward)
            steps_taken = step
            
            log_step(
                step=step,
                action=f"priority={priority_num}",
                reward=reward,
                done=done,
                error=error
            )
            
            if done:
                break
        
        # Calculate score (average reward, already in 0-1 range)
        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score >= 0.5  # Consider success if average reward >= 0.5
        
    except Exception as e:
        print(f"Task error for {task_id}: {e}", file=sys.stderr, flush=True)
        score = 0.0
        success = False
    
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return {
        "task_id": task_id,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
    }


def main() -> None:
    """Run inference on all 3 tasks."""
    if not API_KEY:
        print(
            "No LLM credentials found. Set OPENAI_API_KEY for the default OpenAI path, "
            "or keep using the legacy Hugging Face settings with HF_TOKEN/API_BASE_URL.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
