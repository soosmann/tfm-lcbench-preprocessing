import json
from pathlib import Path

INPUT_PATH = "data/bench_full.json"
OUTPUT_DIR = Path("data/split")

OUTPUT_DIR.mkdir(exist_ok=True)

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

# data is a dict: {dataset_name: dataset_content}
for dataset_name, dataset_content in data.items():
    out_path = OUTPUT_DIR / f"{dataset_name}.json"

    with open(out_path, "w") as f:
        json.dump(dataset_content, f, indent=2)

print(f"Saved {len(data)} files to {OUTPUT_DIR}")
