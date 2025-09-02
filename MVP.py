import os
from openai import OpenAI
import json
import numpy as np
import requests
from bs4 import BeautifulSoup


# ---- Configuratie ----
# API configuratie voor LLM en vector database

api = "ai_specialist_key_1"
api_code = "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"

client = OpenAI(api_key=os.getenv("AI_SPECIALIST_KEY_1", api_code))

# Load cases from JSON file
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

cases = load_jsonl("/Users/carlosalmeida/Downloads/yarado_sales_agent_dummy_zorg_v2_240.jsonl")

# emebed complete cases with; id, task_name, organisation_name, organisation_profile, analysed_tasts, ai_tast_summary


def embed_cases(cases):
    embeddings = []
    for case in cases:
        text_to_embed = case.get("description", "")  #
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text_to_embed
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

case_embeddings = embed_cases(cases)


#  --- agent functionality ---  
class Session: 
    def __init__(self):
        self.context = []
        self.history = []
        self.compliance_check = None
        self.query = None
    
    def add_message(self, role,  message):
        self.history.append({
            "role": role,
            "content": message
        })

    def get_history(self, max_length=10):
        return self.history[-max_length:]
    

def preprocess_question(question):
    jargon = ["DBC", "VECOZO", "HL7", "EPD", "VVT", 'NEN7510", "AVG"']
    for term in jargon:
        question = question.replace(term, f"<{term}> (zorgspecifiek jargon)")
    return question

def get_similar_cases(query_embedding, case_embeddings, cases, top_k=5):
    similarities = []
    for i, embedding in enumerate(case_embeddings):
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
        similarities.append((similarity, cases[i]))
    
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [case for _, case in similarities[:top_k]]

def get_case_details(query, case, case_embeddings):
    case_details = {
        "title": case.get("title", "Geen titel"),
        "description": case.get("description", "Geen beschrijving"),
        "status": case.get("status", "Onbekend"),
        "created_at": case.get("created_at", "Onbekend"),
        "updated_at": case.get("updated_at", "Onbekend")
    }
    
    # Add embedding similarity score
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    
    similarity = np.dot(query_embedding, case_embeddings) / (np.linalg.norm(query_embedding) * np.linalg.norm(case_embeddings))
    case_details["similarity"] = similarity
    
    return case_details

def fetch_compliance_info(topic): 
    if "AVG" in topic or "persoongsgegevens" in topic or "privacy" in topic:
        url = "https://www.autoriteitpersoonsgegevens.nl/nl/onderwerpen/avg-europese-privacywetgeving"
    elif "veiligheid" in topic or "beveiliging" in topic:
        url = "https://www.nen.nl/nen-7510-veiligheid-in-de-zorg/"
    else:
        url = "https://www.rijksoverheid.nl/onderwerpen/gezondheidszorg"

        return url
        

def build_prompt(session, user_question, similar_cases, web_compliance_info): 
    system_promt =(    "Je bent een AI-agent voor automatisering in de zorg. "
    "Beantwoord vragen op basis van bestaande cases uit het zorgdomein, maar denk ook mee over alternatieve oplossingen die nog niet in de dataset staan. "
    "Check en adviseer altijd over actuele compliance-eisen (zoals AVG, NEN7510, GDPR) en vermeld als je aanvullende eisen online gevonden hebt. "
    "Zoek, indien nodig, naar actuele compliance-richtlijnen van het web en geef samenvattingen. "
    "Vertaal altijd zorg- en automatiseringsjargon naar begrijpelijke taal voor niet-IT-professionals.")
    messages = [
        {"role": "system", "content": system_promt}] 
    messages.extend(session.get_history())
    messages.append({"role": "user", "content": user_question})

    cases_text ="\n\n".join([case.get("description", "") for case in similar_cases])
    messages.append({"role": "assistant", "content": f"Hier zijn enkele vergelijkbare cases:\n{cases_text}"})

    if web_compliance_info:
        messages.append({"role": "assistant", "content": f"Compliance-informatie gevonden: {web_compliance_info}"})
    return messages
    
def generate_llm_answer(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()  

# --- Main loop ---
while True:
    user_input = input("Stel je vraag: ")
    if user_input.lower() in ["exit", "quit", "stop"]:
        break
    
    session = Session()
    session.add_message("user", user_input)
    
    preprocessed_question = preprocess_question(user_input)
    
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=preprocessed_question
    ).data[0].embedding
    
    similar_cases = get_similar_cases(query_embedding, case_embeddings, cases)
    
    web_compliance_info = fetch_compliance_info(user_input)
    
    messages = build_prompt(session, user_input, similar_cases, web_compliance_info)
    
    answer = generate_llm_answer(messages)
    
    print(f"Antwoord: {answer}")
    
    session.add_message("assistant", answer)