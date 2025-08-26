from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class chatState(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]


llm = ChatGroq(
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"
)

def chat_node(state : chatState):
    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages" : [response]}

graph = StateGraph(chatState)


graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile()

initial_state = {
    'messages': [HumanMessage(content='What is the capital of india')]
}

print(chatbot.invoke(initial_state))




