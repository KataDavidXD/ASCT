# ASCT-ActorRollout control

This snapshot supplies the actor-continuation control reported in the discussion and Appendix F. Use method identifier `cf_actor`.

Use Python 3.12 in a dedicated training environment for this snapshot. From this directory:

```sh
python -m pip install -e packages/agent-uct
python -m pip install -e .
asct-train --help
```

The public package uses the ASCT name. Internal imports remain `atppo_lab` for compatibility with the frozen result records. Do not install multiple snapshots in the same environment.

Paper runs use the exact ordered QIDs in `../../new_controls/actor_cf/metadata/splits.json`, with `--train-size 2000 --valid-size 400 --test-size 400`, seeds 11, 23 and 37, and three iterations. Historical launcher defaults use 500 validation/test items; supply the paper arguments explicitly. Consult the [repository README](../../README.md) and Appendix A for the full command and learner configuration. The recorded appendix-control metrics specify any method-specific parameters.

Model weights, datasets, indices and a compatible GPU runtime are external inputs. The evaluator requires `wtb[langgraph-sqlite]==0.3.1`. Model and output paths are command-line inputs. Dependency ranges are preserved from the source snapshot; the offline numerical checks at the repository root do not require this training installation.
