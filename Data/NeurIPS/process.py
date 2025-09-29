import pandas as pd

parquet_file = "./neurips2024.parquet"
data = pd.read_parquet(parquet_file)

accepted_papers = data[data["decision"] == "Accept (Published)"][["abstract", "keywords"]]

rejected_papers = data[(data["decision"] == "Reject") | (data["decision"] == "Desk rejected")][["abstract", "keywords"]]

withdrawn_papers = data[data["decision"] == "Withdrawn"][["abstract", "keywords"]]

accepted_papers.to_json("accepted_papers.json", orient="records", lines=True)
rejected_papers.to_json("rejected_papers.json", orient="records", lines=True)
withdrawn_papers.to_json("withdrawn_papers.json", orient="records", lines=True)

print("JSON complete")
