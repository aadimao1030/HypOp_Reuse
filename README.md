# HypOp

Here you can find the code for HypOp, a tool for combinatorial optimization that employs hypergraph neural networks. It is versatile and can address a range of constrained optimization problems.
In the current version, we have included the following problems: graph and hypergraph MaxCut, graph MIS, SAT, and Resource Allocation (see paper for details). To add new problems, add the appropriate loss function in the loss.py file and add the appropriate function in data_reading.py to read your specific dataset. 

#### Install Required Packages

```bash
pip install -r dependency.txt
```

## Experiment Setups

Performance benchmarks were conducted on an Intel Xeon 8358P CPU with a single or multiple NVIDIA RTX 4090 GPUs depending on the task.

---

### CPU Experiments

All CPU experiments are configured as follows. You only need to replace the path in `run.py`:

- **Hypergraph MaxCut Experiment**:
  - Use `configs/Hypermaxcut_syn.json`
  - For SA (Simulated Annealing): Set `"epoch"` in `configs/Hypermaxcut_syn.json` to `1`
  - For Adam: Set `"Adam"` to `true`
  - For bipartite: Set `"mode"` to `"bipartite"`

- **MIS (Maximum Independent Set) Experiment**:
  - Use `configs/maxind_syn_qubo.json`
  - For pignn: Set `"Niter_h"` to `1`

- **Transfer Learning**:
  - From MaxCut to MIS: `configs/maxind_transfer.json`
  - From Hypermaxcut to Hypermincut: `configs/mincut_transfer.json`

---

### Distributed Experiments

For distributed experiments, replace the path in `run_dist.py`:

- **Hypergraph MaxCut Experiment**:
  - Stanford datasets: `configs/dist_configs/maxcut_stanford_for.json`
  - Arxiv datasets: `configs/dist_configs/maxcut_arxiv_for.json`

- **Parallel Multi-GPU Experiments**:
  - Set `"test_mode"` to `"multi_gpu"` in `run_dist.py`

---

### Generalization Experiments

- **MaxClique Problem**:
  - Run `maxclique_training.py`

- **QAP Problem**:
  - Run `QAP_training.py`