# Experience data format

Every rollout is stored continuously from the first autonomous frame. Human intervention data is not
split away from the policy failure that caused it.

```text
episode_YYYYmmdd_HHMMSS_<id>.partial/
  metadata.json
  DO_NOT_ADD_TO_SFT
  events.jsonl
  frames.jsonl
  images/top/00000000.jpg
  images/wrist/00000000.jpg
```

A cleanly finished episode is atomically renamed from `.partial` to `.complete` and gains
`result.json`. A crash leaves a readable `.partial` directory. Run `lingbot-recap audit` to mark and
list recoverable sessions.

Each frame records:

- observation and both camera paths;
- action proposed by LingBot;
- action actually sent to the follower;
- action source (`lingbot_policy` or `human_intervention`);
- control mode and timestamps.

Each event records policy chunks, detector alerts, intervention reason, leader alignment, verified
leader torque state, hand-back, outcome, and runtime errors. This is an RL experience journal. It is
intentionally excluded from existing SFT manifests.

## Multi-policy teacher labels

Offline `lingbot-recap relabel` adds two sidecar files without changing the source journal:

```text
teacher_labels.jsonl
teacher_labels.meta.json
```

Each label row contains the source `frame_index`, exact task, teacher key/checkpoint/server, inference
timing, the first `teacher_action`, the complete 16-step `teacher_chunk`, and the original student action.
The metadata records the labeling stride, allowed action sources, counts, and the
`MULTI_POLICY_ON_POLICY_DISTILLATION_ONLY` usage boundary. An interrupted run leaves
`teacher_labels.partial.jsonl` and resumes from already completed frame indices.

The LeRobot exporter writes `DISTILLATION_DATASET_ONLY` and `distillation_provenance.json` into the
derived dataset. It splits non-contiguous labeled ranges into separate episodes so future-action lookup
cannot cross a human takeover gap.
