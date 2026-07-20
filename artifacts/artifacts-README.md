# artifacts/

Training writes model checkpoints here, organized
by task and model, one per seed:

```
artifacts/
└── {node,link,graph}/
    └── {gnn,mlp}/
        └── weights_seed_XX.pt
```

Everything except this file is git-ignored. Checkpoints are reproducible from a
seed and the locked environment, so they are rebuilt rather than stored.
