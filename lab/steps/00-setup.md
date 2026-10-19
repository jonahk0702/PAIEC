# STEP 00: Server setup and data

**Goal:** a working Python environment on the server and the training tables downloaded.

## Do
1. From the repo root: `python3.11 -m venv .venv && . .venv/bin/activate`
2. `pip install pandas pyarrow numpy scipy scikit-learn huggingface_hub pyyaml pytest sentence-transformers torch`
3. Hugging Face login: `hf auth login`. **Jonah does this**, because the account must have
   access to the gated dataset `aims-foundations/measurement-db`. If it is not done,
   stop and write that in RESULT.md.
4. `python lab/scripts/fetch_tables.py --revision 54074453da7b9911be0f48e4c560be547bc84d21`
5. `python lab/scripts/fetch_tables.py --revision 54074453da7b9911be0f48e4c560be547bc84d21 --tables responses items`
   (about 5 GB, which can take an hour; run it in the background with `nohup`).
6. Count: eligible dirs in `lab/data/inventory.json`, dirs with a `responses.parquet`, dirs
   with an `items.parquet`, and dirs whose responses are all NaN.

## Outputs
`lab/results/STEP-00/RESULT.md` (the counts, Python version, GPU name, free disk) and
`lab/results/STEP-00/manifest.json`.

## Done when
163 eligible, 163 with responses, 163 with items, 4 all-NaN. If any count differs, write
what differs and stop.
