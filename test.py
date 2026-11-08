import json
from scipy.stats import binomtest
a = {json.loads(l)["id"]: json.loads(l)["acc"] for l in open("eval/chat_L7/pubmedqa/predictions.jsonl")}
b = {json.loads(l)["id"]: json.loads(l)["acc"] for l in open("eval/sft_teacher_L7/pubmedqa/predictions.jsonl")}
kd_only = sum(a[i] and not b[i] for i in a); sft_only = sum(b[i] and not a[i] for i in a)
print("KD right only:", kd_only, " SFT right only:", sft_only,
      " p =", binomtest(kd_only, kd_only + sft_only).pvalue)

