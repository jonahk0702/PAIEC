# AIMS lab: start here

This folder is the whole job for the run model. It needs nothing outside this repo.

```
lab/
  START.md      ← you are here (the loop and the prompts)
  STATUS.md     ← which step is next
  SPEC.md       ← the contract: task, data, splits, protocol, metric, models, results format
  steps/        ← one file per step, done in order
  scripts/      ← data download
  paiec_lab/    ← the code (created in step 01)
  results/      ← small results, committed (RESULT.md, manifest.json, runs.jsonl)
  data/, artifacts/  ← big files, git-ignored
```

## The loop

1. **Design** (Opus / Sol, in Jonah's design repo): ideas, papers and experiments,
   turned into a numbered step file here.
2. **Push** this repo.
3. **Run** (Qwen on the server): do the next step, write its results, push.
4. **Review** (Opus): read `results/`, decide what advances, write the next steps.
5. Repeat. Review gates (`waiting-review`) stop Qwen until a reviewer has looked.

## Prompts

**Qwen, first session:**
```
You are the implementation agent for our PAIEC competition entry. Work only inside this repo.
Read lab/START.md, lab/SPEC.md and lab/STATUS.md. Then do the first `todo` step in lab/STATUS.md,
following its file in lab/steps/ exactly. Write its outputs as SPEC §8 says, update STATUS.md,
commit and push. Then stop and tell me in 5 lines what you did and the step's verdict.
Do not start the next step.
```

**Qwen, every time after ("yo"):**
```
Continue the AIMS lab: read lab/STATUS.md and do the next `todo` step (stop if anything is
`waiting-review`). Follow lab/SPEC.md. Commit, push, and give me 5 lines.
```

**Qwen, when a check fails:**
```
Step <NN> failed: <paste the failing check>. Fix only that, rerun the step's checks,
update RESULT.md and STATUS.md, commit and push. Do not change anything else.
```

**Opus / Sol, to review results (in the design repo):**
```
Pull thesis/AIMS/PAIEC. Read lab/STATUS.md, lab/results/runs.jsonl and the RESULT.md files
for steps <NN–MM>. Apply the decision rule in lab/SPEC.md §7. Write
lab/results/STEP-<NN>/REVIEW.md (PASS / fixes), then write the next step files
(lab/steps/<next>.md) and add rows to STATUS.md. Keep each step self-contained:
the run model cannot see the design repo.
```

## Rules for the run model
- One step at a time. Never skip a `waiting-review`.
- Never touch the lockbox benchmarks in `results/splits.json` unless a step says so.
- Never submit to Codabench. Build the zip, check it, and tell Jonah.
- Never commit data, zips, weights, `.env` or tokens.
- If a step is ambiguous, write the question in RESULT.md, set the status to
  `blocked: question`, and stop. Do not guess.
