cd "/data-processing–preliminary/Q5"

# 1) Ensure the broken .venv is not active

deactivate 2>/dev/null || true

# 2) Use uv only (as requested)

uv venv .uvenv
source .uvenv/bin/activate
uv pip install -r requirements.txt

# 3) Verify data files exist

ls -la "/Users/taher/Projects/data-processing–preliminary/Q5/Data"

python train_and_predict.py \
 --train "/Users/taher/Projects/data-processing–preliminary/Q5/Data/train.csv" \
 --test "/Users/taher/Projects/data-processing–preliminary/Q5/Data/test.csv" \
 --out "/Users/taher/Projects/data-processing–preliminary/Q5/notebook/submission.csv"

wc -l "/Users/taher/Projects/data-processing–preliminary/Q5/notebook/submission.csv"
head -5 "/Users/taher/Projects/data-processing–preliminary/Q5/notebook/submission.csv"
