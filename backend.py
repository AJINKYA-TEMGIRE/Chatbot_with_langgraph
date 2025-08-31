from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from ddgs import DDGS
import sqlite3



load_dotenv()

llm = ChatGroq(
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"
)

@tool
def web_search(query: str) -> str:
    """Search the web for any information using DuckDuckGo."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, region="us-en", max_results=5))
        formatted = []
        for r in results:
            formatted.append(f"{r['title']} - {r['href']}\n{r['body']}")
        return "\n\n".join(formatted)

tools = [web_search]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]

def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools) 

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node") 


chatbot = graph.compile(checkpointer=checkpointer)

# out = chatbot.invoke({"messages": [HumanMessage(content = "Please tell todays news about india.")]})
# print(out["messages"][-1].content)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)



