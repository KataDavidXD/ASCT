# ASCT

**Attentive Search over Counterfactual Trees for Credit Assignment in Agentic Reinforcement Learning**

[Paper](https://arxiv.org/abs/2609.35215) · [HTML](https://arxiv.org/html/2609.35215v1) · [Publication scope](PUBLICATION.md)

ASCT uses auxiliary search during training to assign credit to the decisions of a workflow actor. At each visited state, an evaluator compares legal actions from the same recoverable prefix. The frozen actor's action probabilities center the resulting action-value table, and PPO updates the actor's own sampled decisions. Deployment runs the actor alone.

This repository contains the implementation and numerical evidence for **arXiv:2609.35215v1**. Uniform, UCT and cost-aware AgentUCT instantiate the evaluator. The RAG workflow uses WTB to restore shared execution prefixes.

## Start with the retained results

The offline reconstruction requires Python 3.12 and no model inference. From the repository root, create and activate a virtual environment, then run:

```sh
python -m pip install -r requirements-analysis.txt
python tools/verify_public_release.py
python reproduce.py
```

The reconstruction verifies input hashes, recomputes reported results and paired comparisons, checks the auxiliary-token ledger and training credit, and rebuilds eight paper figures. Its outputs are written to `_checks/`, `analysis/`, `analysis_new_controls/` and generated files under `paper_data/`. The optional ninth figure, the appendix evaluator schematic, also requires Poppler's `pdffonts`:

```sh
python paper_data/build_flow_figures.py
```

The scripts operate on retained numerical inputs. Repeating GPU training additionally requires the original models, datasets, indices and a compatible runtime.

## Paper results

HotpotQA final-policy results from Table 2. Trained rows are mean ± sample standard deviation over seeds 11, 23 and 37; Base is fixed.

| Policy | Workflow utility | Official answer F1 | Execution words |
| --- | ---: | ---: | ---: |
| Base | 0.5648 | 0.6954 | 5,348.7 |
| PPO | 0.5651 ± 0.0021 | 0.6913 ± 0.0021 | 5,170.4 ± 76.8 |
| VinePPO | 0.5939 ± 0.0062 | 0.7115 ± 0.0026 | 4,817.7 ± 179.7 |
| ASCT-Uniform | 0.6146 ± 0.0191 | 0.7316 ± 0.0168 | 4,792.2 ± 151.4 |
| ASCT-UCT | 0.6075 ± 0.0241 | 0.7257 ± 0.0260 | 4,841.2 ± 112.8 |
| ASCT-AgentUCT | 0.6187 ± 0.0076 | 0.7266 ± 0.0126 | 4,420.8 ± 213.3 |

Utility is `answer_F1 - 0.1 * execution_words / 4096`. Training uses the repository answer scorer; the table uses the official HotpotQA scorer. Auxiliary search tokens are accounted for separately from workflow utility. The paper's cost comparison includes recorded Qwen environment calls and auxiliary actor scoring; it does not measure total GPU compute or monetary cost.

## Repository map

| Path | Contents and connection to the paper |
| --- | --- |
| `code/main/` | ASCT, PPO and VinePPO; RAG workflow; transfer and component-description runners |
| `code/actor_rollout/` | ASCT-ActorRollout continuation control, reported in the discussion and Appendix F |
| `code/at2po/` | Workflow adaptation of AT²PO, reported in Appendix F |
| `code/probes/` | Frozen-state allocation, budget and cost-phase controls in Appendix E |
| `results/` | Main-task endpoints, component and transfer probes, auxiliary ledgers and compact training-credit records |
| `new_controls/` | Matched predictions, training metrics and protocol evidence for the two appendix controls |
| `paper_data/` | Numerical inputs and figure/table reconstruction scripts |
| `tools/` | Scoring, numerical reconstruction and publication checks |
| `MANIFEST.json` | Exact public file allowlist, byte sizes and SHA-256 hashes |

The three training directories are separate snapshots. Install each in its own environment. Their shared internal import name, `atppo_lab`, and historical result identifiers are retained so that code and recorded artifacts remain compatible. The project and public command use **ASCT**.

## Run the main paper configuration

Use a separate Python 3.12 training environment with suitable CUDA support. Install the evaluator and main implementation from the repository root:

```sh
python -m pip install -e code/main/packages/agent-uct
python -m pip install -e code/main
```

Provide Qwen3-4B-Instruct-2507, E5-base-v2, BGE-M3, Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B, together with the HotpotQA data. The generator and retrieval components stay frozen; the actor uses LoRA. The workflow retrieves within each question's supplied candidate passages.

```sh
asct-train \
  --model-path /path/to/Qwen3-4B-Instruct-2507 \
  --rag-model-root /path/to/paper-components \
  --dataset-root /path/to/hotpotqa \
  --output-dir /path/to/new-run \
  --devices cuda:0 cuda:1 \
  --split-manifest new_controls/actor_cf/metadata/splits.json \
  --train-size 2000 --valid-size 400 --test-size 400 \
  --methods agentuct_atppo uct_atppo uniform_atppo ppo vineppo \
  --seeds 11 23 37 --iterations 3 \
  --search-budget 12 --epochs 2 --ppo-batch-size 8 \
  --search-cost-mode agentuct-Bellman --beta-tok 0.0001 \
  --workflow-cost-lambda 0.1 --execution-cost-reference 4096
```

In PowerShell, put the command on one line or use PowerShell continuation syntax. Device selection must match the available GPU memory and runtime.

The exact ordered QID manifest is essential: the paper's 2,000/400/400 sets are mutually disjoint subsets of the **official training partition**, rather than the official hidden test set. Both held-out cohorts contain 76 easy, 256 medium and 68 hard questions. The historical low-level launcher defaults to 500 validation and test questions and can sample another split if no manifest is supplied; use the explicit paper arguments above. See [training details](code/main/README.md) and the paper's Appendix A.

| Paper label | Recorded method identifier |
| --- | --- |
| ASCT-AgentUCT | `agentuct_atppo` |
| ASCT-UCT | `uct_atppo` |
| ASCT-Uniform | `uniform_atppo` |
| PPO / VinePPO | `ppo` / `vineppo` |
| ASCT-ActorRollout | `cf_actor`, in its separate snapshot |
| AT²PO adaptation | `at2po`, in its separate snapshot |

The training dependency ranges come from the frozen snapshots. Offline reconstruction is checked independently of that GPU environment. The repository contains compact evidence, not model weights, trained checkpoints, full indices or execution databases.

## Code checks

```sh
python -m pip install -r requirements-tests.txt
python check_code.py
```

Each snapshot runs in an isolated test process. Tests that require unavailable training dependencies report skips; these do not establish a successful GPU training run.

## Citation and attribution

```bibtex
@misc{li2026asct,
  title = {ASCT: Attentive Search over Counterfactual Trees for Credit Assignment in Agentic Reinforcement Learning},
  author = {Yang Li and Jinhan Yang and Hai Liu and Di Wan and Xiyu Chen and Zongsi Xu and Tuo Zhou and Sheng Zhong and Sergey Volkov and Ye Luo and Hao Sun},
  year = {2026},
  eprint = {2609.35215},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  doi = {10.48550/arXiv.2609.35215},
  url = {https://arxiv.org/abs/2609.35215}
}
```

Source code is provided under the [MIT license](LICENSE), with included component notices preserved. See [third-party attribution](THIRD_PARTY_NOTICES.md). External models and datasets retain their respective terms.
