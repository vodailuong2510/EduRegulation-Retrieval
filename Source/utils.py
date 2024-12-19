import pandas as pd
import json

def read_csv(path:str) -> pd.DataFrame:
    return pd.read_csv(path)

def convert_to_squad_format(df:pd.DataFrame) -> dict:
    squad_data = {"context": [], "question": [], "answers": [], "yes_no_label": [], "abstractive_answer": [], "document": []}

    for _, row in df.iterrows():
        entry = {
            "title": row["article"] if "article" in row else "unknown",
            "paragraphs": [
                {
                    "context": row["context"],
                    "qas": [
                        {
                            "id": str(row["index"]),
                            "question": row["question"],
                            "answers": [
                                {
                                    "text": row["extractive answer"],
                                    "answer_start": row["context"].find(row["extractive answer"])
                                }
                            ],
                            "is_impossible": False if row["extractive answer"] else True
                        }
                    ]
                }
            ]   
        }

        squad_data["data"].append(entry)

    with open("squad_format_data.json", "w", encoding="utf-8") as f:
        json.dump(squad_data, f, ensure_ascii=False, indent=4)

    return squad_data