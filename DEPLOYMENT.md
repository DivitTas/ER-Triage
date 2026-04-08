# Deploying ER Triage to HuggingFace Spaces

This guide covers deploying the ER Triage environment to HuggingFace Spaces for the OpenEnv Hackathon.

## Prerequisites

1. **HuggingFace Account**: Sign up at https://huggingface.co if you don't have one
2. **HF Token**: Get from https://huggingface.co/settings/tokens (needs write access)
3. **Docker Image Built**: Run `docker build -t er-triage-env:latest -f server/Dockerfile .`

## Method 1: Using OpenEnv CLI (Recommended)

The easiest way to deploy is using the `openenv push` command:

### Step 1: Prepare Environment
```bash
cd /home/divit/Projects/openenv/ER_Triage
export PATH="$HOME/.local/bin:$PATH"
source /home/divit/Projects/openenv/venv/bin/activate
```

### Step 2: Login to HuggingFace
```bash
# If not already logged in
huggingface-cli login
# Paste your HF token when prompted
```

### Step 3: Push to Spaces
```bash
# Basic push (uses username from HF login + env name from openenv.yaml)
openenv push

# Or specify custom repo ID
openenv push --repo-id your-username/er-triage-env

# Make it a private Space
openenv push --repo-id your-username/er-triage-env --private
```

### Step 4: Wait for Build
- Go to https://huggingface.co/spaces/your-username/er-triage-env
- Wait 2-5 minutes for Docker build to complete
- Space status will change from "Building" to "Running"

### Step 5: Test Deployment
```bash
SPACE_URL="https://your-username-er-triage-env.hf.space"

# Test health
curl $SPACE_URL/health

# Test reset
curl -X POST $SPACE_URL/reset -H "Content-Type: application/json"

# View API docs
open $SPACE_URL/docs
```

## Method 2: Manual Git Deployment

If `openenv push` doesn't work, you can deploy manually:

### Step 1: Create New Space on HuggingFace
1. Go to https://huggingface.co/new-space
2. Choose a name (e.g., `er-triage-env`)
3. Select **Docker** as SDK
4. Choose visibility (Public or Private)
5. Click "Create Space"

### Step 2: Clone and Configure
```bash
# Clone the new Space repo
git clone https://huggingface.co/spaces/your-username/er-triage-env
cd er-triage-env

# Copy ER Triage files
cp -r /home/divit/Projects/openenv/ER_Triage/* .

# Ensure README.md has proper YAML frontmatter
# (Should already exist from ER_Triage/README.md)
```

### Step 3: Commit and Push
```bash
git add .
git commit -m "Initial deployment of ER Triage environment"
git push
```

### Step 4: Monitor Build
- Watch build logs at: https://huggingface.co/spaces/your-username/er-triage-env
- Build takes 2-5 minutes
- If build fails, check logs for errors

## Method 3: Using HuggingFace Hub CLI

### Step 1: Install Hub CLI
```bash
pip install huggingface_hub[cli]
huggingface-cli login
```

### Step 2: Create Space
```bash
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id="your-username/er-triage-env",
    repo_type="space",
    space_sdk="docker",
    private=False
)
```

### Step 3: Upload Files
```bash
cd /home/divit/Projects/openenv/ER_Triage

huggingface-cli upload your-username/er-triage-env . . \
    --repo-type space \
    --exclude ".git" --exclude "__pycache__" --exclude "*.pyc" --exclude ".venv"
```

## Troubleshooting

### Build Fails with Import Errors
- Check that `server/app.py` has correct import fallbacks
- Verify all dependencies in `pyproject.toml`
- Test Docker build locally first:
  ```bash
  docker build -t er-triage-test -f server/Dockerfile .
  docker run -p 8000:8000 er-triage-test
  ```

### Space Shows "Application Error"
- Check Space logs for errors
- Verify Dockerfile CMD is correct: `uvicorn server.app:app --host 0.0.0.0 --port 8000`
- Ensure port 8000 is exposed in Dockerfile

### Health Check Fails
- Wait 30 seconds after Space shows "Running"
- Try accessing `/health` endpoint directly
- Check Space logs for startup errors

### WebSocket Connection Issues
- Ensure Space URL uses HTTPS (HF provides this automatically)
- Check browser console for connection errors
- Verify WebSocket endpoint at `/ws`

## Validation

After deployment, run the validation script:

```bash
cd /home/divit/Projects/openenv/ER_Triage
bash validation.py https://your-username-er-triage-env.hf.space .
```

Expected output:
```
✓ HF Space deploys
✓ Responds to ping
✓ OpenEnv spec compliance
✓ Dockerfile builds
✓ Baseline reproduces
✓ 3+ tasks with graders
```

## Space Configuration

Your Space should have these features:

- **URL**: `https://your-username-er-triage-env.hf.space`
- **Endpoints**:
  - `/health` - Health check
  - `/reset` - Start new episode
  - `/step` - Execute action
  - `/state` - Get current state
  - `/docs` - API documentation (Swagger)
  - `/ws` - WebSocket for persistent sessions
  - `/web` - Interactive UI (if enabled in openenv)

## Post-Deployment

1. **Test all 3 tasks**: Verify task_1, task_2, task_3 all work
2. **Run inference remotely**: Test with `ErTriageEnv(base_url="https://...")`
3. **Share Space URL**: Submit to hackathon organizers
4. **Monitor usage**: Check Space logs for any issues

## Tips

- **Private Spaces**: Use `--private` flag if testing before final submission
- **Custom Domain**: HF Spaces supports custom domains in paid tiers
- **Logs**: Always check Space logs if something doesn't work
- **Rebuild**: If Space is stuck, click "Factory reboot" in Settings

## Additional Resources

- OpenEnv Docs: https://github.com/meta-pytorch/openenv
- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- Docker Spaces Guide: https://huggingface.co/docs/hub/spaces-sdks-docker
