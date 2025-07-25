import traceback
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_ollama import ChatOllama


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.0
)


# Incorporating the chat model into a simple node
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# Add the node to Graph
graph_builder.add_node("chatbot", chatbot)  # (unique node name, function or object that will be called)

# Graph direction
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile Graph
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break