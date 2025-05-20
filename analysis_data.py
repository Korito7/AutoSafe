import re
import json
import numpy as np

data_path = "/data/zxy/data/snapshot/public/test_snapshots.json"
data = json.load(open(data_path, encoding="utf-8"))

exp_lens = []
for d in data:
    exp_len = len(d["exps"])
    exp_lens.append(exp_len)

max_exp_len = max(exp_lens)
min_exp_len = min(exp_lens)
avg_exp_len = sum(exp_lens) / len(exp_lens)
median_exp_len = np.median(exp_lens)



print("Max exp len: ", max_exp_len)
print("Min exp len: ", min_exp_len)
print("Avg exp len: ", avg_exp_len)
print("Median exp len: ", median_exp_len)