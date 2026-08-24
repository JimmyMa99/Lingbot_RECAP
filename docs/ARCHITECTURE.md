# Architecture and ownership

## Runtime modules

| Module | Owns | Must not do |
|---|---|---|
| `detectors` | chunk-level state history and alert scores | command either arm |
| `inputs` | keyboard/button events | decide control authority |
| `handoff` | state transitions and the single-writer rule | infer policy actions |
| `hardware` | serial buses, alignment and torque readback | label outcomes |
| `journal` | durable experience records | add data to SFT manifests |
| `policy` | LingBot HTTP requests | write motor commands |
| `runtime` | scheduling modules at control frequency | bypass handoff authority |

## State transitions

```text
AUTO
  | detector alert: hold follower and wait for operator
  | manual Space: explicit takeover request
  v
TAKEOVER_PENDING --R--> AUTO
  | Space
  v
ALIGNING_LEADER
  | align success + all six Torque_Enable read 0
  v
HUMAN --Space--> AUTO

Any alignment/readback/control exception -> FAULT -> preserve partial data and stop.
```

The follower remains position-controlled during handoff. The leader is actively aligned, then
unloaded. Human control is impossible unless torque readback succeeds for every leader motor.

## Training boundary

The collector records all policy, intervention and outcome data. A separate future pipeline will:

1. validate complete episodes;
2. derive sparse returns from outcome and subtask events;
3. train a value model;
4. compute timestep advantages;
5. construct advantage-conditioned LingBot examples.

No component in this repository's collection runtime mutates existing LingBot SFT manifests.
