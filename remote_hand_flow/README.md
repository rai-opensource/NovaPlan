# Remote Hand-Flow Service

This folder contains NovaPlan's remote service for hand-centric grounding. The
public service is named **hand flow**; its current reconstruction backend is
upstream [HaMeR](https://github.com/geopavlakos/hamer).

- `bootstrap_hand_flow_host.sh`: one-command setup, launch, and health check.
- `setup_hand_flow_host.sh`: lower-level one-time backend setup.
- `launch_hand_flow_server.sh`: day-to-day service launch.
- `hand_flow_server.py`: FastAPI `/predict` service.

The request/response API, metric calibration metadata, and output handling are
NovaPlan code. The detector, crop-inference, and weak-camera conversion
sequence in `hand_flow_server.py` is adapted from HaMeR's `demo.py` at the
pinned revision. Its retained MIT terms are in
[HAMER_LICENSE.md](HAMER_LICENSE.md).

The paper deployment used an RTX A6000 for this service. The service may run on
the workstation or a remote GPU host.

## Install

First download [`MANO_RIGHT.pkl`](https://mano.is.tue.mpg.de/), then run from the repository root:

```bash
MANO_RIGHT_SOURCE=/path/to/MANO_RIGHT.pkl \
  ./remote_hand_flow/bootstrap_hand_flow_host.sh
```

The bootstrap creates or updates the Pixi environment, clones HaMeR and ViTPose,
downloads the public HaMeR checkpoint bundle, places `MANO_RIGHT.pkl`, launches
the service, and checks `/health`.

The default backend checkout is:

```text
remote_hand_flow/.external/hamer
```

Expected backend assets are:

```text
$HAMER_DIR/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
$HAMER_DIR/_DATA/data/mano/MANO_RIGHT.pkl
```

The setup is resumable. Once these assets exist, rerunning bootstrap does not
require `MANO_RIGHT_SOURCE` and does not redownload completed files:

```bash
./remote_hand_flow/bootstrap_hand_flow_host.sh
```

For setup without launch:

```bash
START_HAND_FLOW_SERVER=0 ./remote_hand_flow/bootstrap_hand_flow_host.sh
```

## Launch

For day-to-day use after setup:

```bash
export PORT=8080
pixi run -e hand-flow-host launch-hand-flow-server
```

The launcher replaces an existing hand-flow process and clears the configured
port by default. It writes `logs/hand_flow_server.log`.

Check health:

```bash
curl http://127.0.0.1:8080/health
```

The response identifies `service=novaplan-hand-flow` and `backend=hamer`.

## Connect

When the service and workstation are on the same machine:

```bash
export NOVAPLAN_HAND_FLOW_SERVER_URL=http://127.0.0.1:8080/predict
```

For an SSH-accessible remote host:

```bash
ssh -L 8080:127.0.0.1:8080 USER@HAND_FLOW_HOST
export NOVAPLAN_HAND_FLOW_SERVER_URL=http://127.0.0.1:8080/predict
```

For Kubernetes, substitute the deployed pod and namespace and run the commands below from the local workstation:

```bash
kubectl port-forward pod/<hand-flow-pod> -n <namespace> 8080:8080
export NOVAPLAN_HAND_FLOW_SERVER_URL=http://127.0.0.1:8080/predict
```

To expose the service directly on a trusted network instead, launch with
`HOST=0.0.0.0` and set the URL to that host.

## Backend Options

Override the upstream checkout location:

```bash
export HAMER_DIR=/path/to/hamer
./remote_hand_flow/bootstrap_hand_flow_host.sh
```

The default HaMeR detector is `vitdet`. Use the smaller backend when needed:

```bash
export HAMER_BODY_DETECTOR=regnety
pixi run -e hand-flow-host launch-hand-flow-server
```

## Verify

With both the object-flow and hand-flow services running, execute the live installation
check from the workstation:

```bash
export NOVAPLAN_OBJECT_FLOW_SERVER_URL=http://127.0.0.1:7001
export NOVAPLAN_HAND_FLOW_SERVER_URL=http://127.0.0.1:8080/predict
pixi run -e local-planning-full verify-hand-flow-server
```

The check requests a fresh hand mask and 3D tracking bundle from the object-flow
service, checks the hand-flow service health endpoint, sends the recorded
rollout, and verifies the reconstructed hand output under
`runs/verification/hand_flow/`. The `hamer_outputs` name is retained only
because those files are direct HaMeR reconstruction artifacts.

## API Contract

The service exposes `GET /health` and `POST /predict`. `/predict` accepts a
NumPy-serialized, base64-encoded RGB frame array plus optional camera
intrinsics. The response includes `real_meshes` records containing `frame_idx`,
`hand_id`, `is_right`, `vertices`, `faces`, and optional semantic finger
landmarks. The workstation client calibrates these meshes to the metric flow
scene before using them for hand-centric grounding.
