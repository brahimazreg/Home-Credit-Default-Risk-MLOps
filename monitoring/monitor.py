import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset



# logs
rows = []
with open("logs/predictions.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))

logs = pd.DataFrame(rows)
current_data = pd.json_normalize(logs["input"])

# reference (SAFE for CI)
reference_data = pd.read_csv("data/reference_sample.csv")

report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=reference_data,
    current_data=current_data
)

report.save_html("monitoring/drift_report.html")

print("Done")