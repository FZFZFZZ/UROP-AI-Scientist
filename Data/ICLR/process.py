import pandas as pd

parquet_files = ["./iclr2024_accepted.parquet", "./iclr2025.parquet"]

all_data = pd.DataFrame()

for file in parquet_files:
    data = pd.read_parquet(file)
    all_data = pd.concat([all_data, data], ignore_index=True)

accepted_papers = all_data[all_data["decision"] == "Accept (Published)"][["abstract", "keywords"]]

rejected_papers = all_data[(all_data["decision"] == "Reject") | (all_data["decision"] == "Desk rejected")][["abstract", "keywords"]]

withdrawn_papers = all_data[all_data["decision"] == "Withdrawn"][["abstract", "keywords"]]

accepted_papers.to_json("accepted_papers.json", orient="records", lines=True)
rejected_papers.to_json("rejected_papers.json", orient="records", lines=True)
withdrawn_papers.to_json("withdrawn_papers.json", orient="records", lines=True)

print("JSON files generated successfully.")
