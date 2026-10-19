# STEP 02: Review gate (do nothing; wait)

The harness code is reviewed by Opus or Sol before any experiment counts. Set STATUS
row 02 to `waiting-review`, commit, push and stop.

The reviewer reads `lab/paiec_lab/protocol.py`, `metrics.py`, `splits.py`, the tests
and `trace.md`. It then writes `lab/results/STEP-02/REVIEW.md` with either `PASS` or a
numbered fix list. If there are fixes: do exactly those, rerun the tests, and set row 02
back to `waiting-review`. Continue to step 03 only after `PASS`.
