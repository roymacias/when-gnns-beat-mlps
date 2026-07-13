# data/

Regenerated, not versioned. `make data` downloads the benchmark datasets here
(PyTorch Geometric creates one subfolder per dataset, each with its own `raw/`
and `processed/`) and materializes the seeded splits under `splits/`:

```
data/
├── Cora/           # node classification
├── Computers/      # link prediction (Amazon Computers)
├── NCI1/           # graph classification
└── splits/
    ├── link/       # seed_XX.pt: message graph + val/test pairs
    └── graph/      # seed_XX.pt: stratified train/val/test indices
```

Everything except this file is git-ignored. Node classification uses Cora's
canonical masks, so no split file is materialized for it.
