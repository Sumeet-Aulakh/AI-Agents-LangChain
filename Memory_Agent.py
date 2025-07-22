import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print(f"State: {state}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

if os.path.exists("logging.txt"):
    print("Reading Conversation History...")
    with open("logging.txt", "r") as f:
        for line in f.readlines():
            conversation_history.append(line)


user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]

    user_input = input("Enter: ")

with open("logging.txt", "w") as f:

    f.write("Your conversation history:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
    f.write("End of conversation history.\n")

print("Conversation Saved to logging.txt")