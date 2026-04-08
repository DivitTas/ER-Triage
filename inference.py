"""
ER Triage Inference Script
===================================
Runs an LLM agent through the ER Triage environment for all 3 tasks.

Environment Variables (from .env):
    API_BASE_URL   - The API endpoint for the LLM (default: HuggingFace router)
    MODEL_NAME     - The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       - Your HuggingFace API token

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import re
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from ER_Triage import ErTriageEnv, ErTriageAction, TriagePriority
from ER_Triage.server import ErTriageEnvironment

# Load environment variables from .env
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("IMAGE_NAME")  # Optional: for Docker-based env

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


def get_triage_decision(client: OpenAI, obs) -> int:
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
        return parse_priority(response)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 3  # Default to URGENT on error


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
            priority_num = get_triage_decision(client, obs)
            priority = TriagePriority(priority_num)
            action = ErTriageAction(priority=priority)
            
            # Take step
            obs = env.step(action)
            
            reward = obs.reward
            done = obs.done
            error = None
            
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
        print(f"[DEBUG] Task error: {e}", flush=True)
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
        print("[ERROR] HF_TOKEN not set. Please set it in your .env file.", flush=True)
        return
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    results = []
    for task_id in TASKS:
        print(f"\n{'='*50}", flush=True)
        print(f"Running {task_id}...", flush=True)
        print(f"{'='*50}\n", flush=True)
        
        result = run_task(client, task_id)
        results.append(result)
    
    # Summary
    print(f"\n{'='*50}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*50}", flush=True)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{status} {r['task_id']}: score={r['score']:.2f}, steps={r['steps']}", flush=True)
    
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.2f}", flush=True)


if __name__ == "__main__":
    main()

