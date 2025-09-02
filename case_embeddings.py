import json
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Zorg dat key uit .env komt

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

cases = load_jsonl("/Users/carlosalmeida/Downloads/yarado_sales_agent_dummy_zorg_v2_240.jsonl")

def embed_cases(cases):
    embeddings = []
    skipped_cases = []
    for idx, case in enumerate(cases):
        text_to_embed = " ".join([
            str(case.get("task_name", "")),
            str(case.get("organisation_name", "")),
            str(case.get("organisation_profile", "")),
            str(case.get("analysed_tasks", "")),
            str(case.get("ai_task_summary", "")),
        ])

        ai_task_summary = case.get("ai_task_summary", "")
        if not ai_task_summary or ai_task_summary == "Automation outline coming soon":
            skipped_cases.append(case.get("id", idx))
            continue

        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text_to_embed
        )
        # Sla op: (id, embedding vector) pair
        embeddings.append({
            "id": case.get("id", idx),
            "embedding": response.data[0].embedding
        })
    return embeddings, skipped_cases

embeddings, skipped = embed_cases(cases)

# Sla de embeddings op voor snelle document search
with open("case_embeddings.json", "w", encoding="utf-8") as out:
    json.dump(embeddings, out, indent=2)

print(f"{len(embeddings)} cases ge-embed. {len(skipped)} cases overgeslagen (niet compleet).")
