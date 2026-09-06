# CSE445 Assignment #3 — Autonomous Local LLM Machine Learning Agent

## Deliverables
- `ml_tools.py` — baseline + three advanced ML tools
- `react_agent.py` — Ollama ReAct controller with self-correction
- `benchmark_runner.py` — 3-algorithm × 2-dataset CV benchmark
- `benchmark_results.json` — generated local benchmark results
- `requirements.txt`
- `execution_log.txt` — reproducible validation log
- `TECHNICAL_REPORT.pdf` — 3–5 page technical report

## WSL2 setup
Run in Windows PowerShell (Administrator):
```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```
Then in Ubuntu:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv curl build-essential git
mkdir -p ~/cse445_agent && cd ~/cse445_agent
python3 -m venv venv
source venv/bin/activate
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
pip install --upgrade pip
pip install -r requirements.txt
python benchmark_runner.py
python react_agent.py
```

## Important reproducibility note
The included benchmark results and validation log were generated in the submission-building Python environment. They validate the ML implementation, but they do **not** claim that Ollama/WSL2 was executed by this ChatGPT session. For the strongest submission, run the final `react_agent.py` command in your own WSL2 terminal and append its terminal output to `execution_log.txt`.

## ReAct protocol
The controller follows Thought → Action → Action Input → Observation and stops at Final Answer. Tool errors are fed back as observations so the local model can correct invalid parameters and retry.

## Advanced tools
1. GridSearchCV for SVC and Decision Tree.
2. StandardScaler + PCA feature reduction.
3. PyTorch MLP with BatchNorm, Dropout and ReduceLROnPlateau.
