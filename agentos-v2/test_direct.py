import asyncio
import sys
import traceback
from langchain_core.messages import HumanMessage
from agent import graph, AgentState

async def main():
    print("Initializing test run...")
    multi_agent_system = graph.compile()
    
    initial_state = {
        "messages": [HumanMessage(content="Hi!")],
        "next": "Supervisor",
        "iterations": 0,
    }
    config = {"configurable": {"thread_id": "test-thread-id"}}
    
    try:
        print("Invoking graph...")
        result = await multi_agent_system.ainvoke(initial_state, config=config)
        print("Graph invocation success!")
        print("Result:", result)
    except Exception as e:
        print("Graph invocation failed!", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
