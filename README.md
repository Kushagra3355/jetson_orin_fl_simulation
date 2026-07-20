# AiProff Jetson Orin Federated BiPAP Simulation

This package turns the lecture use case into a complete classroom laboratory. Each NVIDIA Jetson Orin board represents one synthetic patient node. It generates breathing records, stores them locally, trains a small model and transmits only a protected model update to a coordinator.

> **Safety boundary:** This is an educational simulation using synthetic data. It is not medical device software. Never connect it to a BiPAP device, patient sensor, clinical record system or treatment workflow.

## What students can prove

1. Every node has a different patient distribution, so the data are non IID.
2. Local training happens where the data are generated.
3. The federated API rejects raw breathing fields.
4. Pairwise masks hide individual client updates from the coordinator in the classroom secure aggregation mode.
5. The global model improves over rounds and returns to every node.
6. The centralized baseline can achieve strong accuracy, but only after copying all raw training rows to one place.
7. Federated learning improves data minimization and reduces raw data exposure. It does not automatically guarantee better accuracy.

## Project map

```text
common/                 Linear model and privacy helpers
edge/                   Synthetic BiPAP generator and Jetson patient node
server/                 Federated coordinator and static dashboard server
dashboard/              HTML, CSS and JavaScript live display
comparison/             Centralized, local only and federated comparison
tools/                  Configuration generator and preflight check
scripts/                Jetson setup and launch commands
tests/                  Privacy contract and secure mask tests
config/                 Server and node configuration examples
```

## Prerequisites

* NVIDIA Jetson Orin family board with a supported JetPack installation
* Python 3.10 or newer
* Network connectivity between the Jetson nodes and coordinator
* About 200 MB free storage for the virtual environment and generated data

The simulation is CPU light and does not require CUDA, TensorRT, PyTorch or a Node.js runtime. The dashboard JavaScript runs in the browser and is served by Python.

Current NVIDIA JetPack downloads and supported Orin releases are listed at:

https://developer.nvidia.com/embedded/jetpack/downloads

Older supported JetPack releases, including JetPack 6.2.2 for Orin, are listed at:

https://developer.nvidia.com/embedded/jetpack-archive

Do not reflash a working lab board only for this demonstration. Confirm the installed release with your lab administrator.

## One time setup on every Jetson

Copy this folder to each board. From the project root run:

```bash
chmod +x scripts/*.sh
./scripts/setup_jetson.sh
```

The script installs Python virtual environment support, creates `.venv`, installs NumPy and runs the preflight check.

Manual equivalent:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/preflight.py
```

Optional Jetson checks:

```bash
cat /etc/nv_tegra_release
sudo nvpmodel -q
tegrastats
```

Use the lab approved power profile. The simulation does not need MAXN mode.

## Fastest rehearsal on one computer

This command generates four different synthetic patients and compares local only, centralized, secure federated and secure federated with teaching differential privacy:

```bash
source .venv/bin/activate
python -m comparison.run_all_in_one
```

Results are written to `results/comparison.json` and `results/comparison.csv`. Generated raw CSV files remain under `data/<client_id>/`.

## Multi board Jetson lab

### 1. Choose the coordinator

One Jetson or instructor laptop can be the coordinator. Find its LAN address:

```bash
hostname -I
```

Assume the coordinator address is `192.168.1.50`.

### 2. Generate private node configurations

Run on the coordinator:

```bash
source .venv/bin/activate
python tools/generate_client_configs.py \
  --clients jetson01 jetson02 jetson03 jetson04 \
  --server-url http://192.168.1.50:8000 \
  --privacy-mode secure_dp
```

This creates `config/clients/jetson01.json` through `jetson04.json` and updates `config/server.json`.

Copy only the matching client file to each board. Do not give one node every other node's configuration because the pairwise secrets are meant to be separated.

### 3. Start the coordinator

```bash
./scripts/run_server.sh --reset
```

Open the classroom dashboard:

```text
http://192.168.1.50:8000
```

The JSON API is also visible at:

```text
http://192.168.1.50:8000/api/status
```

### 4. Start one patient on each Jetson

On Jetson 1:

```bash
./scripts/run_node.sh config/clients/jetson01.json --watch
```

Use the corresponding JSON file on Jetsons 2, 3 and 4. A round completes when every secure aggregation participant submits. The global model then advances and watch mode automatically joins the next round.

### 5. What to point out live

* The raw file path printed by each node is local to that board.
* The node prints `raw rows sent 0` for every round.
* The dashboard reports `Raw patient rows at server: 0`.
* The coordinator waits for all secure aggregation clients so pairwise masks cancel.
* Weighted local MAE should generally fall over early rounds.
* Each profile remains different, demonstrating non IID data.

Stop clients and coordinator with `Ctrl+C`.

## Centralized comparison

Run:

```bash
python -m comparison.centralized_baseline
```

The baseline joins all client training rows before training. This often provides strong accuracy, but the coordinator can read every synthetic respiratory rate, tidal volume, leak and oxygen saturation value.

| Question | Centralized learning | Federated secure mode |
|---|---|---|
| Where are raw training rows stored? | Coordinator | Original Jetson only |
| Raw rows sent over network | All training rows | Zero |
| Can coordinator read patient fields? | Yes | No |
| Can coordinator read an individual update? | Not applicable | No in the complete mask group |
| Can it learn across patients? | Yes | Yes |
| Accuracy guaranteed to be better? | No | No |
| Main advantage | Simple pooled optimization | Data minimization and reduced raw data exposure |

## Privacy modes

* `weights_only` sends an unmasked model update. Raw rows remain local, but individual updates may leak information.
* `secure` adds pairwise masks that cancel in the sum. All configured clients must participate.
* `dp` clips the client update and adds small teaching noise. This illustrates the privacy and utility tradeoff.
* `secure_dp` combines the two classroom mechanisms.

The secure aggregation helper is intentionally understandable rather than production grade. Real deployments require authenticated transport, robust cryptography, dropout recovery, key management, authorization, audit controls and a formally reviewed differential privacy budget.

## Run automated checks

```bash
python -m unittest discover -s tests -v
```

The checks confirm that synthetic data can train a model, update payloads omit patient fields, pairwise masks cancel and the coordinator rejects a raw oxygen saturation field.

## Troubleshooting

**Coordinator unavailable**

Confirm all boards can reach the coordinator IP and TCP port 8000. Check local firewall rules and make sure `server_url` does not use `127.0.0.1` on remote boards.

**A secure round never completes**

Every client in `expected_clients` must submit the same round with the same privacy mode. Start the missing board or reset the coordinator and restart all nodes.

**Masks did not cancel**

Regenerate the client configurations and recopy only the correct configuration to each node. All nodes must use the same generated configuration set.

**NumPy installation fails**

Verify the JetPack release, network access and system clock. As a lab administrator, install the distribution package `python3-numpy` if your environment blocks Python package downloads, then create the virtual environment with `python3 -m venv --system-site-packages .venv`.

## Official references

* NVIDIA JetPack SDK downloads: https://developer.nvidia.com/embedded/jetpack/downloads
* NVIDIA JetPack archive: https://developer.nvidia.com/embedded/jetpack-archive
* NVIDIA Jetson power and performance documentation: https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html
* Python virtual environments: https://docs.python.org/3/library/venv.html
* NumPy documentation: https://numpy.org/doc/stable/
