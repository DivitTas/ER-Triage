---
title: ER Triage Environment Server
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - healthcare
---

# ER Triage Environment

An OpenEnv reinforcement learning environment simulating emergency room patient triage. Agents must assign priority levels (1-5) to patients based on vital signs while managing limited critical care beds and time pressure from patient deterioration.

## Quick Start

The simplest way to use the ER Triage environment is through the `ErTriageEnv` class:

```python
from ER_Triage import ErTriageAction, ErTriageEnv, TriagePriority

try:
    # Create environment from Docker image
    env = ErTriageEnv.from_docker_image("er-triage-env:latest")

    # Reset to start a new episode
    result = env.reset()
    obs = result.observation
    print(f"Patient {obs.patient_id}: {obs.chief_complaint}")
    print(f"Vitals: BP {obs.systolic_bp}/{obs.diastolic_bp}, HR {obs.heart_rate}, SpO2 {obs.oxygen_saturation}%")

    # Triage patients
    while obs.patients_remaining > 0:
        # Assign priority based on vitals (your agent logic here)
        if obs.oxygen_saturation < 85 or obs.systolic_bp < 80:
            priority = TriagePriority.CRITICAL
        elif obs.heart_rate > 120 or obs.respiratory_rate > 24:
            priority = TriagePriority.EMERGENT
        else:
            priority = TriagePriority.URGENT
        
        result = env.step(ErTriageAction(priority=priority))
        obs = result.observation
        print(f"Assigned P{priority} → Reward: {result.reward:.3f}")

finally:
    # Always clean up
    env.close()
```

That's it! The `ErTriageEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From ER_Triage directory
export DOCKER_BUILDKIT=1
docker build -t er-triage-env:latest -f server/Dockerfile .

# Test the container
docker run -d -p 8000:8000 --name er-triage-test er-triage-env:latest
curl http://localhost:8000/health
docker logs er-triage-test
docker stop er-triage-test && docker rm er-triage-test
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Tasks
Three difficulty levels with varying patient counts and severity distributions:
- **task_1**: 5 patients, balanced severity (easiest)
- **task_2**: 10 patients, moderate critical cases
- **task_3**: 15 patients, high critical/emergent load (hardest)

### Action
**ErTriageAction**: Assign triage priority to current patient
- `priority` (TriagePriority) - Priority level 1-5:
  - **1 - CRITICAL**: Immediate life threat (cardiac arrest, severe trauma)
  - **2 - EMERGENT**: High risk, needs rapid care (chest pain, stroke)
  - **3 - URGENT**: Stable but needs timely care (high fever, fractures)
  - **4 - LESS_URGENT**: Can wait 1-2 hours (sprains, mild pain)
  - **5 - NON_URGENT**: Minor issues (cold symptoms, refills)

### Observation
**ErTriageObservation**: Patient vitals and resource state
- `patient_id` (int) - Unique patient identifier
- `systolic_bp`, `diastolic_bp` (int) - Blood pressure (mmHg)
- `heart_rate` (int) - Heart rate (bpm)
- `respiratory_rate` (int) - Respiratory rate (breaths/min)
- `temperature` (float) - Body temperature (Celsius)
- `oxygen_saturation` (int) - SpO2 percentage
- `chief_complaint` (str) - Patient's main complaint
- `patients_remaining` (int) - Patients left in queue
- `critical_beds_available` (int) - Available critical care beds (0-2)
- `current_patient_wait_time` (int) - Steps this patient has waited

### Reward Structure
Rewards are in 0.0-1.0 range based on:
- **Correct triage** (+0.8-1.0): Priority matches patient severity
- **Under-triage** (0.0-0.4): Assigning too low priority to critical patients
- **Over-triage** (0.4-0.6): Wasting critical beds on stable patients
- **Time penalties**: Patients deteriorate if they wait too long

### RL Challenge
This environment presents a true sequential decision problem:
- **Resource constraints**: Only 2 critical beds; assigning P1 uses a bed for 4 steps
- **Time pressure**: Critical patients deteriorate if not treated quickly
- **Trade-offs**: Must balance bed availability with patient urgency

## Advanced Usage

### Connecting to an Existing Server

If you already have an ER Triage environment server running, you can connect directly:

```python
from ER_Triage import ErTriageEnv, ErTriageAction, TriagePriority

# Connect to existing server
env = ErTriageEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = env.reset()
result = env.step(ErTriageAction(priority=TriagePriority.URGENT))
```

Note: When connecting to an existing server, `env.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from ER_Triage import ErTriageAction, ErTriageEnv, TriagePriority

# Connect with context manager (auto-connects and closes)
with ErTriageEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation
    
    # Process all patients in queue
    while obs.patients_remaining > 0:
        # Your triage logic here
        priority = TriagePriority.URGENT
        result = env.step(ErTriageAction(priority=priority))
        obs = result.observation
        print(f"Patient triaged → Reward: {result.reward:.3f}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    ErTriageEnvironment,  # Pass class, not instance
    ErTriageAction,
    ErTriageObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from ER_Triage import ErTriageAction, ErTriageEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with ErTriageEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(ErTriageAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the ER_Triage directory
python3 -c "
from server.ER_Triage_environment import ErTriageEnvironment
from models import ErTriageAction, TriagePriority

env = ErTriageEnvironment(task_id='task_1')
state = env.reset()
print(f'Initial patient: {state.observation.chief_complaint}')

for i in range(5):
    state = env.step(ErTriageAction(priority=TriagePriority.URGENT))
    print(f'Step {i+1}: Reward={state.reward:.3f}, Remaining={state.observation.patients_remaining}')
"
```

This verifies that:
- Environment resets correctly for all tasks
- Step executes triage actions properly
- Resource constraints work (beds, deterioration)
- Rewards are in 0.0-1.0 range

### Running Locally

Run the server locally for development:

```bash
# Using uv (recommended)
cd ER_Triage
uv sync
uv run server --port 8000

# Or using uvicorn directly
uvicorn server.app:app --reload --port 8000
```

Test the server:
```bash
# Check health
curl http://localhost:8000/health

# Test reset
curl -X POST http://localhost:8000/reset

# View API docs
open http://localhost:8000/docs
```

## Project Structure

```
ER_Triage/
├── .dockerignore         # Docker build exclusions
├── .env                   # Environment variables (not committed)
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # ErTriageEnv client
├── models.py              # Action and Observation models
├── grader.py              # Episode scoring (0.0-1.0 range)
├── inference.py           # LLM agent runner for hackathon
├── validation.py          # Pre-submission validation script
└── server/
    ├── __init__.py        # Server module exports
    ├── ER_Triage_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```

## Hackathon Inference

Run the LLM agent through all 3 tasks:

```bash
# Set up environment variables in .env
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_token_here

# Run inference
uv run python inference.py
```

The script outputs structured logs for validation:
```
[START] task=task_1 env=er_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=priority:2 reward=0.85 done=false error=null
[END] success=true steps=5 score=0.82 rewards=0.85,0.90,0.75,0.80,0.82
```
