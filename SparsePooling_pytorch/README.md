
# Efficient, GPU-ready pytorch implementation of the SparsePooling framework

Please use the virtual environment (conda) for this codebase. Create from dependency file (env.yml) via:
```bash
    bash ./setup_dependencies.sh
```

Depending on your OS, different env.yml will be used by setup_dependencies.sh. 
Environment was built (on respective OS) using:

```bash
    conda create -n sparsepooling_pytorch python=3
    installing all the packages..
    conda env export > whateverpath/env.yml
```

Code base (not algorithm) inspired by:
https://github.com/lpjiang97/sparse-coding/
and
https://github.com/loeweX/Greedy_InfoMax



