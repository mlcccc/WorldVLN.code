# Remote Simulator Workflow

The remote_sim path keeps policy inference on the training host and delegates only environment reset, action execution, and rendering to the simulator host.

## Service Endpoints

The simulator-side service is implemented by runtime/infinity_tsformer_client.py in service mode. It exposes:

- GET /health
- POST /reset
- POST /step_actions

## Start the Service

On the simulator machine:

```bash
python runtime/infinity_tsformer_client.py \
  --mode service \
  --host 0.0.0.0 \
  --port 8765 \
  --task_json_root /path/to/UAV-Flow-Eval/test_jsons
```

## Reverse Port Forwarding

If the simulator machine cannot be reached directly from the training host, create a reverse tunnel from the simulator machine to the training host:

```bash
ssh -R 18765:127.0.0.1:8765 user@training-host
```

Then validate from the training host:

```bash
curl --noproxy '*' http://127.0.0.1:18765/health
```

## StageA Settings

Use these environment variables for StageA:

```bash
UAVFLOW_STAGEA_ROLLOUT_BACKEND=remote_sim
UAVFLOW_SIMULATOR_BASE_URL=http://127.0.0.1:18765
UAVFLOW_TASK_JSON_ROOT=/path/to/UAV-Flow-Eval/test_jsons
STAGEA_NPROC=1
```

STAGEA_NPROC should remain 1 unless the simulator service is extended to support multiple concurrent sessions.
