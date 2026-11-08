"""Step 5: lifecycle energy and break-even (reviewers R1/R3: does distillation pay for itself?).

    python break_even.py --spec configs/break_even.yaml

For each option:  E(N) = E_datagen + E_search + E_train + N * e_request
A student breaks even against the baseline after  N* = (E_once,student - E_once,base) / (e_base - e_student)
requests (never, if it is not cheaper per request). Energy values are read from the files the
other scripts write: data/teacher/<tag>/manifest.json, runs/<run>/summary.json, bench/<name>.json.
Prints a table, writes results/break_even.csv and a cumulative-energy plot.
"""
import argparse
import os

import yaml

from common import read_json

KWH = 3.6e6


def per_request(bench_file, setting):
    b = read_json(bench_file)
    if "rate" in setting:
        return next(r for r in b["online"] if r["offered_rps"] == setting["rate"])["j_per_request"]
    if setting.get("offline"):
        return b["offline"]["j_per_request"]
    return next(r for r in b["hf"] if r["batch_size"] == setting["batch_size"])["mean"]["j_per_request"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default="results/break_even.csv")
    args = ap.parse_args()
    with open(args.spec) as fh:
        spec = yaml.safe_load(fh)
    setting = spec["setting"]
    opts = []
    for o in spec["options"]:
        datagen = sum(read_json(o["datagen"]["manifest"])[k]["energy"]["energy_j"] for k in o["datagen"]["keys"]) \
            if o.get("datagen") else 0.0
        search = sum(read_json(f)["energy"]["energy_j"] for f in o.get("search", []))
        train = read_json(o["train"])["energy"]["energy_j"] if o.get("train") else 0.0
        opts.append({"name": o["name"], "per_request_j": per_request(o["bench"], setting),
                     "one_time_j": datagen + search + train, "datagen_kwh": datagen / KWH,
                     "search_kwh": search / KWH, "train_kwh": train / KWH})
    base = next(o for o in opts if o["name"] == spec["baseline"])
    for o in opts:
        saving = base["per_request_j"] - o["per_request_j"]
        o["break_even_requests"] = None if saving <= 0 else max(0.0, (o["one_time_j"] - base["one_time_j"]) / saving)
        for rpd in spec.get("requests_per_day", [1e4, 1e5]):
            n = o["break_even_requests"]
            o[f"days@{int(rpd)}/day"] = None if n is None else n / rpd

    import pandas as pd
    df = pd.DataFrame(opts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_max = 2 * max([o["break_even_requests"] or 0 for o in opts] + [1e4])
    xs = [n_max * i / 100 for i in range(101)]
    for o in opts:
        plt.plot(xs, [(o["one_time_j"] + x * o["per_request_j"]) / KWH for x in xs], label=o["name"])
    plt.xlabel("requests served")
    plt.ylabel("cumulative GPU energy (kWh)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200)


if __name__ == "__main__":
    main()
