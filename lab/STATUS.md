# STATUS

Qwen: do the **first row whose status is `todo`**, unless a row above it is
`waiting-review` (then stop). After finishing a step, set its status to `done` (or
`blocked: <reason>`), fill in the date, commit and push.

| Step | File | Status | Date | Note |
| --- | --- | --- | --- | --- |
| 00 | steps/00-setup.md | todo | | Jonah does `hf auth login` |
| 01 | steps/01-harness.md | done | 19 Oct | Written by Opus; 17 tests pass; M2 beats M1 |
| 02 | steps/02-review-gate.md | waiting-review | 19 Oct | Sol reviews the harness code |
| 03 | steps/03-baselines.md | todo | | Hand zips to Jonah |
| 04 | steps/04-rasch-update.md | todo | | |
| 05 | steps/05-mcni.md | todo | | |
| 06 | steps/06-graph-and-transfer.md | todo | | Then `waiting-review` |

Target dates: 00–01 Tue 20 Oct · 02 review Tue night · 03 Wed 21 · 04 Thu 22 ·
05–06 Fri 23. Steps 07+ are written after the review of 03–06. Deadline Fri 30 Oct AoE.
