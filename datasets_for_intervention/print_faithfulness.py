import sys
import json

path = sys.argv[1]

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for structure_type, metrics in data["metrics"]["faithfullness"].items():
    print(f"\n{structure_type}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: mean = {value['mean']}, std = {value['std']}")
