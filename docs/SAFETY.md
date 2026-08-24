# Safety contract

1. Exactly one process owns `/dev/ttyACM0` and `/dev/ttyACM1`.
2. Detection only pauses the follower and alerts the operator. It never moves the leader.
3. Space is an explicit authorization to align the leader.
4. During alignment the operator must release the leader and keep hands clear.
5. The follower stays torque-enabled and holds its measured pose throughout handoff.
6. Human control is granted only after all six leader motors read back `Torque_Enable == 0`.
7. A failed write, failed readback, overload, stale camera, or non-finite action enters a fault path.
8. A physical power switch or emergency stop remains authoritative over software.

Do the first test with no object, low workspace obstruction, and a second person at the power switch.
