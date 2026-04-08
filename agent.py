import operator
import os
from functools import partial
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from tools import add_database_task, python_execution_tool, research_tool

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


class Route(BaseModel):
    next: Literal["CodexEngine", "ResearchAgent", "DBAgent", "FINISH"] = Field(
        description="Select the specialized agent for the task."
    )


async def supervisor_node(state: AgentState):
    """Generalist Supervisor: Routes to specialists and handles final synthesis."""
    prompt = SystemMessage(content="Route to Codex for code, Research for data, or DBAgent for storage.")
    routing_llm = llm.with_structured_output(Route)
    response = await routing_llm.ainvoke([prompt] + list(state["messages"]))
    return {"next": response.next if isinstance(response, Route) else "FINISH"}


async def worker_node(state: AgentState, agent_llm, name: str):
    """Worker node with the Anti-Parrot Synthesis pass."""
    response = await agent_llm.ainvoke(state["messages"])
    new_messages = [response]
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_map = {
                "add_database_task": add_database_task,
                "research_tool": research_tool,
                "python_execution_tool": python_execution_tool
            }
            result = await tool_map[tool_call["name"]].ainvoke(tool_call["args"])
            new_messages.append(ToolMessage(content=str(result), name=tool_call["name"], tool_call_id=tool_call["id"]))

        final_response = await agent_llm.ainvoke(list(state["messages"]) + new_messages)
        new_messages.append(final_response)
    return {"messages": new_messages}

workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("CodexEngine", partial(worker_node, agent_llm=llm.bind_tools([python_execution_tool]), name="Codex"))
workflow.add_node("ResearchAgent", partial(worker_node, agent_llm=llm.bind_tools([research_tool]), name="Research"))
workflow.add_node("DBAgent", partial(worker_node, agent_llm=llm.bind_tools([add_database_task]), name="DB"))

workflow.add_edge("CodexEngine", "Supervisor")
workflow.add_edge("ResearchAgent", "Supervisor")
workflow.add_edge("DBAgent", "Supervisor")
workflow.add_conditional_edges("Supervisor", lambda x: x["next"], {"CodexEngine": "CodexEngine", "ResearchAgent": "ResearchAgent", "DBAgent": "DBAgent", "FINISH": END})
workflow.set_entry_point("Supervisor")

graph = workflow  # Export the uncompiled graph for main.py to attach checkpointers
workflow = graph.compile()

__all__ = ["graph", "AgentState"]
