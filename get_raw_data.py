import os
import tarfile
import pandas as pd

def process_tar_file(tar_path, dataset_path, label, enron_number):
    data = []
    with tarfile.open(tar_path, "r:gz") as tar:
        file_names = [member.name for member in tar.getmembers() if member.name.startswith(f"{dataset_path}/{label}/") and member.isfile()]
        for file_name in file_names:
            with tar.extractfile(file_name) as file:
                lines = file.read().decode("latin1").split("\n")
                subject = lines[0].replace("Subject: ", "", 1) if lines[0].startswith("Subject: ") else None
                data.append({
                    "subject": subject,
                    "ham/spam": label,
                    "count": 0,
                    "filename": os.path.basename(file_name),
                    "enron": enron_number,
                    "data": '\n'.join(lines[1:])
                })
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=["subject", "ham/spam", "count", "filename", "enron","data"])

    # Count the number of emails with the same subject and ham/spam label
    df["count"] = df.groupby(["subject", "ham/spam"])["subject"].transform("count")


    # !!! Deprecated, this might cause data loss
    # Drop duplicates based on subject and ham/spam label, keeping only the first occurrence
    # df.drop_duplicates(subset=["subject", "ham/spam"], keep="first", inplace=True)

    return df

def main():
    # Get the path to the "resource" folder
    resource_path = "resource"  

    # Process enron 1 to 6
    datasets = range(1, 7)
    dfs = [
        process_tar_file(
            os.path.join(resource_path, f"enron{dataset_number}.tar.gz"),
            f"enron{dataset_number}",
            label,
            dataset_number
        )
        for dataset_number in datasets
        for label in ["ham", "spam"]
    ]

    # Concatenate the DataFrames
    df = pd.concat(dfs, ignore_index=True)

    if not os.path.exists("output"):
        os.makedirs("output")

    # Save to CSV
    df.to_json("output/enron_email_data.json", orient="records")

if __name__ == "__main__":
    main()
