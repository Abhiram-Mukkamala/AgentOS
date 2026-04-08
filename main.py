import uuid
import os
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from agent import graph, AgentState

app = FastAPI(title="AgentOS Backend")
memory = MemorySaver()
multi_agent_system = graph.compile(checkpointer=memory)


class ChatRequest(BaseModel):
    prompt: str
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


@app.post("/chat")
async def chat_with_agents(request: ChatRequest):

    if request.prompt.strip().lower() in ["reset", "clear", "new user"]:
        return {"response": "🧠 Memory Wiped. Starting fresh session.", "thread_id": str(uuid.uuid4())}

    config: RunnableConfig = {"configurable": {"thread_id": request.thread_id}}
    initial_state: AgentState = {"messages": [HumanMessage(content=request.prompt)], "next": "Supervisor"}

    try:
        final_state = await multi_agent_system.ainvoke(initial_state, config=config)
        return {"response": str(final_state["messages"][-1].content), "thread_id": request.thread_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found in container root</h1>", status_code=404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
