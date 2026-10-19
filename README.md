# Auxiliary collision prediction: reproducibility materials

Companion code and recorded data for **Output Ordering and Loss Balance in
Auxiliary Collision Prediction for Robot Navigation**.

The study compares output ordering and loss balance in collision readouts on
six frozen critic representations. It reports original broadcast inputs, an
aligned-action sensitivity, and a separate fresh-task follow-up (Phase B).
Navigation experiments evaluate the corresponding complete training systems.
Score consistency, predictive quality and navigation outcomes are separate
endpoints; the results do not establish a navigation benefit from ordering.

## Download and reproduce

Download the data packages from [release v1.0.0](https://github.com/CUDDLIU/auxiliary-collision-prediction/releases/tag/v1.0.0).
Python 3.9+ and NumPy 1.22+ suffice for frozen-readout numerical replay.

| Resource | Contents | Entry point |
|---|---|---|
| Online Resource 1 | Detailed methods, numerical results and interpretation | PDF |
| Online Resource 2 | Navigation, training and common-input records | `python3 reproduce.py` inside the archive; shared-input command below |
| Online Resource 3 | Original broadcast and aligned sensitivity scores and bootstrap arrays | Commands below |
| Online Resource 4 | Six frozen Q1 exports, 840 original episodes, source code, reference heads and offline neural replay | See its preserved README and `PUBLIC_RELEASE_NOTICE.md` |
| Online Resource 5 | Phase B: 1,260 episodes, 24 selected heads, 480 epoch heads, scores, specifications and diagnostics | Command below |

Unpack Resources 3 and 5 beside this repository's files:

```sh
python3 -m pip install -r requirements.txt
python3 reproduce_readouts.py --study Online_Resource_3/broadcast --out reproduced_original.json
python3 reproduce_readouts.py --study Online_Resource_3/aligned --out reproduced_aligned.json
python3 reproduce_readouts.py --study Online_Resource_5_Phase_B/study --out reproduced_phase_b.json
```

Each command recalculates all four cells, simple effects and interactions from
24 saved score files, verifies common sample/label ordering, and checks the
reported percentile intervals against saved bootstrap arrays. It imports the
unchanged scientific metric functions in `analyze_frozen_probe.py`. No neural
fitting, simulator execution or new bootstrap resampling occurs.

For the common-input diagnostic, run inside Resource 2:
```sh
python3 scripts/recompute_shared_metrics.py --results-dir data/shared_evaluation/results
```
Resource 2 numerical analysis uses the Python standard library. Optional figure
rendering needs matplotlib; its README documents the individual scripts.

## Neural reconstruction and scope

Resource 4 includes the existing offline runner for reconstructing features and
refitting the original broadcast/aligned heads from six Q1 encoders and 840
episodes. Its exact replay environment is Linux, PyTorch 2.4.1+cu121, NumPy
1.23.5 and a compatible CUDA device. See the archive's environment instructions.
The archived anonymous adapter has not itself undergone a fresh neural replay.

Phase B numerical replay is directly runnable. Its scientific producer and
follow-up adapter are supplied for the feature, fitting and selection definitions;
the historical supervisor uses private launch bindings and is not a portable
CLI. A turnkey new Phase B neural refit is not claimed in this release.
Raw episodes, selected/epoch heads and the shared frozen encoders are supplied.
Original SAC training, behaviour-actor checkpoints, ROS-Gazebo installation and
fresh simulator collection are outside this recorded-data release.

## Code map

- `reproduce_readouts.py`: portable numerical replay of saved predictions.
- `analyze_frozen_probe.py`: unchanged metric, contrast and bootstrap definitions.
- `frozen_risk_probe.py`: unchanged original feature/fitting/selection producer.
- `frozen_followup.py`: historical Phase B adapter, including aligned histories
  and validation geometry-deletion analysis; source for inspection.
- `portable_readout.py`: Resource 4 adapter; execute in its complete package.
- `SAC.py`: retained network and loss definitions; supporting files are in Resource 4.

`proper` in source identifiers denotes ordinary, unweighted BCE. AP is grouped-score,
non-interpolated average precision. Raw log loss is calculated from saved logits.
Strict ordering violations use all score vectors; other metrics use the stated masks.
Inference averages metrics over six encoders and jointly resamples source identities
for encoder/actor roles and whole tasks within maps.

## Provenance and release terms

Numeric arrays retain their recorded bytes. Phase B JSON machine-path labels are
redacted in a documented public derivative; its `DERIVATION.json` distinguishes
historical hashes from the new distribution hashes. Public packages include
checksums. A checksum verifies bytes, not scientific validity or preregistration.

This public release was authorized by the authors on 19 October 2026. Historical
confidential/preparation wording inside byte-preserved records describes their
original status; see `PUBLIC_RELEASE_NOTICE.md`. No repository-wide open-source
license has been selected. Copyright and third-party license notices remain in
force; public availability does not grant additional rights beyond applicable terms.
No DOI is asserted.
