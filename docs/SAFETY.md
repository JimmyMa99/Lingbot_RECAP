# Safety contract

1. Exactly one process owns `/dev/ttyACM0` and `/dev/ttyACM1`.
2. Detection only pauses the follower and alerts the operator. It never moves the leader.
3. Button 1 (or the first Space press) is explicit authorization to align the leader.
4. During alignment the operator must release the leader and keep hands clear.
5. The follower stays torque-enabled and holds its measured pose throughout handoff.
6. Button 2 (or the second Space press) is required before unloading the leader. Human control is
   granted only after all six leader motors read back `Torque_Enable == 0`.
7. A failed bus write/readback, camera read failure, or non-finite action enters a fault path.
8. This version does not yet provide an independent motor-current/temperature overload monitor. Do
   not run unattended or treat the two-button keyboard as an emergency stop.
9. A physical power switch or emergency stop remains authoritative over software.

Do the first test with no object, low workspace obstruction, and a second person at the power switch.
