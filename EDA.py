import json

def check_incomplete_cases(jsonl_path, required_fields):
    incomplete_cases = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            case = json.loads(line)
            missing = [field for field in required_fields if not case.get(field)]
            if missing:
                incomplete_cases.append({
                    "index": idx,
                    "id": case.get("id", ""),
                    "task_name": case.get("task_name", ""),
                    "organisation_name": case.get("organisation_name", ""),
                    "missing_fields": missing
                })
    return incomplete_cases

# Stel in wat je minimaal vereist per case:
required = ["id", "task_name", "organisation_name", "organisation_profile", "analysed_tasks", "ai_task_summary"]

cases_path = "/Users/carlosalmeida/Downloads/yarado_sales_agent_dummy_zorg_v2_240.jsonl"
result = check_incomplete_cases(cases_path, required)

print(f"Aantal incomplete cases: {len(result)}\n")
print("Overzicht:")
for case in result:
    print(
        f"Case #{case['index']}  |  id: {case['id']}\n"
        f"  Taak: {case['task_name']}\n"
        f"  Organisatie: {case['organisation_name']}\n"
        f"  Ontbrekende velden: {', '.join(case['missing_fields'])}\n"
        "---------------------------------------------"
    )
