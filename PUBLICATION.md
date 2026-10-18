# Publication scope

This public release is scoped to [arXiv:2609.35215v1](https://arxiv.org/abs/2609.35215v1), submitted on 28 September 2026.

The release contains the paper's ASCT implementation, comparison implementations, workflow and component integrations, reported appendix controls, and compact evidence needed to reconstruct the reported numbers. Its source is the frozen code/data supplement associated with that manuscript, identified by SHA-256 `ed3bcb0634604c59fe803d03a1a5a166edbc044eb4f2d31f86c7b1eac04fea88`.

The source snapshots preserve the paper's search, actor sampling, actor-weighted credit, PPO updates, prompts, scoring rules and numerical inputs. Publication edits update project names, descriptions and attribution, normalize source line endings, and remove unused learner branches, unrelated static-workflow utilities and repository-import audit scripts. An unused learner configuration field was removed from the corresponding retained metadata. Repeated per-question evaluation rows were removed from the six training-metrics files; their scalar summaries and optimizer/search ledgers remain intact, and final matched predictions are retained separately. Retained observations were not recalculated or perturbed. Internal `atppo_lab` imports and recorded method identifiers remain compatible with the evidence.

The repository starts with a fresh Git history. Private development branches, later experiments, manuscript drafts, review notes, credentials, machine configuration, original execution caches and checkpoint archives are outside this release. Models and upstream datasets are obtained separately under their own terms. The paper remains available on arXiv; manuscript drafts and source bundles are not duplicated here.

`MANIFEST.json` is the exact publication allowlist. It records the byte size and SHA-256 of every public file except itself. The generated `.gitignore` admits only these paths, and `tools/verify_public_release.py` checks both their contents and Git's tracked file list. Update the manifest and ignore allowlist deliberately when changing a public file. Generated analysis outputs remain local.

Offline reconstruction verifies retained results and figures. It does not rerun GPU training, certify a new CUDA environment, or supply the separately retained full ActorRollout and AT²PO training archives. The original dependency ranges are preserved; model/runtime changes can affect a new training run.
