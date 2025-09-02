from dotenv import load_dotenv
import os
import numpy as np
from typing import  List, Annotated, Sequence, Dict, Any    
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool   
from langchain.chat_models import init_chat_model 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from googleapiclient.discovery import build
from case_embeddings import embed_cases, load_jsonl
import requests
from bs4 import BeautifulSoup
import httpx
from prompts import PromptTemplates
intent_detection_prompt = PromptTemplates.intent_detection_prompt
document_search_prompt = PromptTemplates.document_search_prompt
web_search_prompt = PromptTemplates.web_search_prompt
compliance_check_prompt = PromptTemplates.compliance_check_prompt
synthesize_results_prompt = PromptTemplates.synthesize_results_prompt

load_dotenv()



# load document embeddings
cases = cases = load_jsonl("/Users/carlosalmeida/Downloads/yarado_sales_agent_dummy_zorg_v2_240.jsonl")
document_embeddings, skipped = embed_cases(cases)


# agent state
class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    user_input : str | None 
    tool_output : Dict[str, str] | None
    intent : str | None
    status : str | None 
    iteration_count:  int | None 
    merge_results_RAG: Dict[str, str] | None
    synthesize_results: Dict[str, Any] | None   



# LLM
llm = init_chat_model("gpt-4o", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")    

#########
# Tools #
#########

@tool
def document_search(query: str, top_k: 5) -> List[Dict]:
    """This tool searches for similar cases in the JSON document that is provided and embedded in case_embedding.py. 
    It looks for similair organisations / RPA solutions and if AI is already used in the automation. 
    After looking for similair cases this tool will provide a sumary that the agent can use in answering the question or to ask questions to the user to explicit the case more. 
    
    """
    query_embedding = embedding_model.embed_query( query)

    similarities = []
    for i, emb in enumerate(document_embeddings):
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        similarities.append((sim, cases[i]))

    similarities.sort(reverse=True, key=lambda x: x[0])
    top_cases = [case for _, case in similarities[:top_k]]
    
    return top_cases

@tool
def google_search(query: str, num_results: int = 5): 
    """Performs a Google search and returns the top results. So the agent can sumarize the information and look into it further.""" 
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    res = service.cse().list(
        q=query,
        cx=GOOGLE_CSE_ID,
        num=num_results
    ).execute()
    
    results = []
    if 'items' in res:
        for item in res['items']:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
    return results

# @tool
# def bing_search(query: str):
#     pass

KeyWords_AVG_GDPR = ["AVG", "GDPR", "privacy", "data protection", "persoonsgegevens", "verwerking", "datalek", "toestemming", "gegevensbescherming", "verantwoordelijke", "verwerker", "recht op inzage", "recht op vergetelheid", "belangenafweging", "privacy by design", "security breach", "DPO (Data Protection Officer)", "informatieplicht", "gegevensbescherming impact assessment (DPIA)", "conflict mediation", "toezicht", "sancties", "cookiebeleid", "beveiligingsmaatregelen"]
KeyWords_ISO_27001 = ["ISO 27001", "information security", "ISMS", "risk management", "security controls", "confidentiality", "integrity", "availability", "asset management", "access control", "cryptography", "physical security", "incident management", "business continuity", "compliance", "internal audit", "management review", "security policy", "supplier relationships", "human resources security"]
KeyWords_NEN_7510 = ["NEN 7510", "healthcare information security", "patient data", "confidentiality", "integrity", "availability", "risk management", "security controls", "compliance", "internal audit", "management review", "security policy", "supplier relationships", "human resources security"]

@tool
def avg_check(query: str) -> str:
    """Checks the AVG compliance for the given query. AVG is the General Data Protection Regulation (GDPR) in the Netherlands.
    So this function retrieves relevant information from the official AVG website. Mostly focussed on Dutch legislation."""
    url = "https://autoriteitpersoonsgegevens.nl/nl/zelf-doen/avg-english"
    response = requests.get(url)
    if response.status_code != 200:
        return "Can not retrieve AVG information."

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    relevant = [p.get_text() for p in paragraphs if any(keyword in p.get_text().lower() for keyword in KeyWords_AVG_GDPR)]
    snippet = "\n".join(relevant[:5])
    return f"AVG Info:\n{snippet}"


@tool
def EU_GDPR_Check(query: str) -> str:
    """Checks the EU GDPR compliance for the given query. This function retrieves relevant information from the official GDPR website.
    So this function is focused on European legislation."""
    url = "https://gdpr-info.eu/"
    response = requests.get(url)
    if response.status_code != 200:
        return "Can not retrieve GDPR information."

    soup = BeautifulSoup(response.text, "html.parser")
    
    paragraphs = soup.find_all("p")
    relevant = [p.get_text() for p in paragraphs if any(keyword in p.get_text().lower() for keyword in KeyWords_AVG_GDPR)]
    snippet = "\n".join(relevant[:5])
    return f"GDPR Info:\n{snippet}"


@tool
def iso_check(query: str) -> str:
    """Checks the ISO 27001 compliance for the given query. This function retrieves relevant information from the official ISO website.
    This certification is focused on information security management systems (ISMS).
    """
    url = "https://www.iso.org/iso-27001-information-security.html"
    with httpx.Client() as client:
        response = client.get(url)
        if response.status_code != 200:
            return "Can not retrieve ISO information."
        
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        relevant = [p.get_text() for p in paragraphs if any(keyword in p.get_text().lower() for keyword in KeyWords_ISO_27001)]
        snippet = "\n".join(relevant[:5])
        return f"ISO Info:\n{snippet}"

@tool
def nen_check(query: str) -> str:
    """Checks the NEN 7510 compliance for the given query. This function retrieves relevant information from the official NEN website.
    """
    url = "https://www.nen.nl/nen-7510-2020-nl-2020-12-01.htm"
    response = requests.get(url)
    if response.status_code != 200: 
        return "Can not retrieve NEN information."
    
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    relevant = [p.get_text() for p in paragraphs if any(keyword in p.get_text().lower() for keyword in KeyWords_NEN_7510)]  
    snippet = "\n".join(relevant[:5])
    return f"NEN Info:\n{snippet}"




#########
# Nodes #
#########

def user_input_node(state: AgentState) -> dict:
    """Extracts the user input from the agent state. Secures the context in a session."""
    user_input = state.get("user_input")
    if not user_input:
        return {}
    human_message = llm.invoke(user_input)
    messages = state.get("messages", [])
    messages.append(human_message)
    return {"messages": messages}


def intent_detection_node(state: AgentState) -> dict:
    """Detects the intent of the user message. Determines the appropriate action to take."""
    messages = state.get("messages", [])
    user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not user_message:
        return {"intent": None}

    prompt = intent_detection_prompt(user_message.content)
    print("Intent Detection Prompt:", prompt)

    response = llm.invoke(prompt)
    print("Intent Detection Response:", response.content)
    intent = response.content.strip().lower()
    return {"intent": intent}


def document_search_node(state: AgentState) -> dict:
    """Searches for relevant documents based on user input."""
    query = state.get("user_input", "")
    if not query:
        return {}
    top_cases = document_search.invoke({"query": query, "top_k": 5})


    summary = "\n\n".join([f"{case['task_name']} (Organisation: {case.get('organisation_name', '')})" for case in top_cases])

    prompt = document_search_prompt(summary)
    print("Document Search Prompt:", prompt)
    response = llm.invoke(prompt)    
    print("Document Search Response:", response.content)
    tool_output = state.get("tool_output", {}) or {}
    tool_output["document_search"] = response.content

    state["tool_output"] = tool_output

    return state


def web_search_node(state: AgentState) -> dict:
    """Performs a web search based on user input."""
    query = state.get("user_input", "")
    if not query:
        return {}
    
    results = google_search.invoke(query) 

    summary = "\n".join([f"{item['title']}: {item['link']}" for item in results]) if results else "Geen webresultaten gevonden."
    
    prompt = web_search_prompt(summary)
    print("Web Search Prompt:", prompt)
    response = llm.invoke(prompt)
    
    tool_output = state.get("tool_output", {}) or {}
    tool_output["web_search"] = response.content
    print("Web Search Response:", response.content)
    state["tool_output"] = tool_output
    return state


def compliance_check_node(state: AgentState) -> dict:
    query = state.get("user_input", "")
    if not query:
        return {}

    avg_info = avg_check.invoke(query)
    gdpr_info = EU_GDPR_Check.invoke(query)
    iso_info = iso_check.invoke(query)
    nen_info = nen_check.invoke(query)

    combined = f"AVG:\n{avg_info}\n\nGDPR:\n{gdpr_info}\n\nISO 27001:\n{iso_info}\n\nNEN 7510:\n{nen_info}"

    prompt = compliance_check_prompt(combined)
    print("Compliance Prompt:", prompt)
    response = llm.invoke(prompt)
    print("Compliance Response:", response.content)

    tool_output = state.get("tool_output", {}) or {}
    tool_output["compliance_check"] = response.content

    state["tool_output"] = tool_output
    return state


def merge_results_RAG_node(state: AgentState) -> dict:
    tool_output = state.get("tool_output", {}) or {}

    doc_search = tool_output.get("document_search", "")
    web_search = tool_output.get("web_search", "")
    compliance_check = tool_output.get("compliance_check", "")


    combined_text = (
        f"Document Search Results:\n{doc_search}\n\n"
        f"Web Search Results:\n{web_search}\n\n"
        f"Compliance Check Results:\n{compliance_check}"
    )

    # Sla op in state voor later gebruik
    state["merge_results_RAG"] = combined_text
    print("Merged RAG results:", combined_text)

    return state


def synthesize_results_node(state: AgentState) -> dict:
    combined_context = state.get("merge_results_RAG", "")
    user_question = state.get("user_input", "")

    if not combined_context or not user_question:
        return state

    prompt = synthesize_results_prompt(combined_context, user_question)
    print("Prompt synthesis:", prompt)
    response = llm.invoke(prompt)
    print("Response synthesis:", response.content)
    state["synthesize_results"] = response.content

    return 


#########
# graph #
#########
graph = StateGraph(AgentState)

# add nodes
graph.add_node("user_input", user_input_node)
graph.add_node("intent_detection", intent_detection_node)
graph.add_node("document_search", document_search_node)
graph.add_node("web_search", web_search_node)
graph.add_node("compliance_check", compliance_check_node)
graph.add_node("merge_results_RAG", merge_results_RAG_node)
graph.add_node("synthesize_results", synthesize_results_node)

# edges zonder condition
graph.add_edge(START, "user_input")
graph.add_edge("user_input", "intent_detection")
graph.add_edge("intent_detection", "document_search")
graph.add_edge("intent_detection", "web_search")
graph.add_edge("intent_detection", "compliance_check")

# verder vaste edges
graph.add_edge("document_search", "merge_results_RAG")
graph.add_edge("web_search", "merge_results_RAG")
graph.add_edge("compliance_check", "merge_results_RAG")

graph.add_edge("merge_results_RAG", "synthesize_results")
graph.add_edge("synthesize_results", END)

app = graph.compile()

############
# programm # 
############

def run_chatbot():
    state = {
        "messages": [],
        "user_input": None,
        "tool_output": {},
        "intent": None,
        "status": None,
        "iteration_count": 0,
        "merge_results_RAG": None,
        "synthesize_results": None,
    }
    print("Chatbot gestart. Typ 'exit' om te stoppen.")

    while True:
        user_input = input("\nWat is je vraag: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        state["user_input"] = user_input
        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)
        # Haal synthese output (of geef fallback)
        answer = result.get("synthesize_results")
        if answer:
            print("\n=== ANTWOORD ===")
            print(answer)
        else:
            # Als synthese_results leeg is, print laatste AIMessage
            messages = result.get("synthesize_results", [])
            if messages:
                print("\n=== ANTWOORD ===")
                print(messages[-1].content)
            else:
                print("\nGeen antwoord gegenereerd.")

run_chatbot()