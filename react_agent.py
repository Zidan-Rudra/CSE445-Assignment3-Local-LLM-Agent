"""
CSE445 Assignment #3 - Local Ollama ReAct controller.
The controller is deliberately defensive: malformed JSON, unknown tools, exceptions,
and repeated failures are returned to the local LLM as observations so it can retry.
"""
import json, re, time, requests
from ml_tools import AVAILABLE_TOOLS

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = """You are an Autonomous Machine Learning Assistant running locally.
Available tools:
- load_dataset_summary(dataset_name)
- train_sklearn_model(dataset_name, model_type, test_size=0.2)
- hyperparameter_tuning(dataset_name, model_type='svc')
- feature_reduction(dataset_name, n_components=2)
- train_pytorch_regularized(dataset_name, hidden_dim=32, epochs=50, lr=0.01, dropout=0.25)

Use exactly:
Thought: ...
Action: tool_name
Action Input: {"key":"value"}
After observations, continue until enough evidence exists.
Finish with:
Thought: ...
Final Answer:
...
Never invent tool observations."""

def query_local_llm(prompt: str) -> str:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1, "stop": ["Observation:"]}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "")

def _parse_action(text):
    am = re.search(r"Action:\s*([A-Za-z0-9_]+)", text)
    im = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if not (am and im): return None, None
    try: return am.group(1), json.loads(im.group(1))
    except json.JSONDecodeError: return am.group(1), None

def run_agent_loop(user_query: str, max_iterations: int = 8):
    prompt = f"{SYSTEM_PROMPT}\nUser Query: {user_query}\n"
    traces, failures = [], 0
    for step in range(1, max_iterations+1):
        started = time.perf_counter()
        output = query_local_llm(prompt)
        latency = time.perf_counter() - started
        traces.append({"step":step, "latency_s":round(latency,3), "llm_output":output})
        if "Final Answer:" in output:
            return {"status":"completed","steps":step,"failures":failures,"traces":traces}
        tool, kwargs = _parse_action(output)
        if not tool:
            failures += 1
            obs = "Observation: Invalid action format. Retry using Action and Action Input JSON."
        elif kwargs is None:
            failures += 1
            obs = "Observation: Error parsing Action Input as JSON. Retry with valid JSON."
        elif tool not in AVAILABLE_TOOLS:
            failures += 1
            obs = f"Observation: Tool '{tool}' not recognized. Select a listed tool."
        else:
            try:
                result = AVAILABLE_TOOLS[tool](**kwargs)
                obs = f"Observation: {result}"
            except (TypeError, ValueError, FloatingPointError) as exc:
                failures += 1
                # Self-healing hints intentionally go back to the model instead of hard-coding
                # a hidden correction.
                obs = f"Observation: Tool execution error: {type(exc).__name__}: {exc}. Re-check parameters, shapes, and finite loss; retry."
            except Exception as exc:
                failures += 1
                obs = f"Observation: Unexpected tool error: {type(exc).__name__}: {exc}. Retry with corrected parameters."
        prompt += output + "\n" + obs + "\n"
    return {"status":"max_iterations","steps":max_iterations,"failures":failures,"traces":traces}

if __name__ == "__main__":
    q = ("Analyze breast_cancer, train a Random Forest and a regularized PyTorch MLP, "
         "compare their accuracies, and recommend the better model.")
    print(json.dumps(run_agent_loop(q), indent=2))
