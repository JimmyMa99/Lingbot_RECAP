# Multi-policy on-policy distillation

This repository implements the data and routing layer needed to distill several task-specialist LingBot
policies into one generalist LingBot policy. The first implementation is deliberately compatible with
LingBot's existing continuous flow-matching action head.

## Why this is not a direct copy of LLM MOPD

LLM MOPD routes every student-generated token prefix to a domain teacher and optimizes a divergence
between teacher and student token probabilities. LingBot emits continuous 16-step action chunks with a
flow-matching head and does not expose an autoregressive action-token log probability. Applying an LLM
MOPD trainer such as NeMo-RL or verl would therefore optimize the language model interface rather than
the SO-101 action distribution.

The practical first stage uses the same on-policy state distribution but a continuous-action target:

1. the student executes a rollout and RECAP stores every observation before the student action;
2. after the rollout, the exact task string routes each stored state to one frozen specialist teacher;
3. the teacher's first action at each student-visited state becomes the distillation action label;
4. contiguous labeled states are exported as a LeRobot v3 dataset;
5. the normal LingBot `L1_fm` trainer mixes this dataset with replay demonstrations.

This is an on-policy DAgger/RLDG-style continuous-action approximation to MOPD. It removes teacher-rollout
exposure bias without requiring an unsafe synchronous teacher call in the robot control loop. A later
stage can replace the first-action target with flow-vector or probability-flow divergence once the
action head exposes a stable scoring API.

## Safety and data boundaries

- Relabeling is offline and never writes motor commands.
- Routing is exact after whitespace normalization. Unknown or duplicate task strings are fatal errors.
- By default only frames whose `action_source` is `lingbot_policy` are labeled. Human takeover states are
  excluded unless `--include-human-states` is explicitly supplied.
- Source RECAP experiences retain `DO_NOT_ADD_TO_SFT`. Exported datasets are separately marked
  `DISTILLATION_DATASET_ONLY` and include `distillation_provenance.json`.
- The exporter refuses to overwrite an existing dataset directory.

## Teacher registry

Copy `configs/multi_policy_teachers.example.json` to a local, machine-specific file and set one server and
checkpoint per specialist. Multiple aliases may point to one teacher, but a task cannot point to two
teachers.

```bash
lingbot-recap validate-teachers \
  --teacher-registry configs/multi_policy_teachers.local.json
```

The teachers do not need to remain loaded together. `relabel` initializes only the teacher selected for
the episode's task. For limited GPU capacity, label one task while its teacher is served, stop that server,
then serve the next teacher and continue.

## Offline teacher labeling

Label all complete student rollouts at the source 30 Hz rate:

```bash
lingbot-recap relabel \
  --teacher-registry configs/multi_policy_teachers.local.json \
  --experience-root /home/mzm/lerobot_data/recap_experience
```

For a fast smoke test, label only eight student frames:

```bash
lingbot-recap relabel \
  --teacher-registry configs/multi_policy_teachers.local.json \
  --episode /path/to/episode.complete \
  --max-frames 8
```

Labeling is crash-resumable through `teacher_labels.partial.jsonl`. Completion atomically produces:

```text
teacher_labels.jsonl
teacher_labels.meta.json
```

Each row preserves the full teacher chunk for analysis and uses `teacher_action` (the first action) for
the initial training dataset.

## Export to LeRobot

```bash
lingbot-recap export-distill \
  --experience-root /home/mzm/lerobot_data/recap_experience \
  --output-root /home/mzm/lerobot_data/mopd_teacher_labeled_v1 \
  --repo-id mzm/lingbot_mopd_teacher_labeled_v1
```

Filtering can create gaps around human interventions. The exporter splits gaps into independent episodes
so LingBot never constructs a future action chunk across a control-authority transition.

## LingBot training recipe

Use the exported dataset as one entry in a normal `multi` training manifest and mix it with the original
clean demonstrations. A conservative starting mixture is one distillation sample for every one or two
replay samples. Keep the same robot config, action/state normalization and camera mapping used by the
teachers.

The first experiment should initialize the student from the common ancestor of all specialists, keep
the specialist teachers frozen, and compare all three real-robot task success rates against:

1. mixed SFT only;
2. parameter averaging;
3. multi-policy on-policy distillation plus replay.

Do not call the result exact reverse-KL MOPD until a teacher/student continuous-action scoring API and a
flow-distribution divergence have been implemented and validated.
