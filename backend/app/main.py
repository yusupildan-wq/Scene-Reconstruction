from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import jobs, projects

app = FastAPI(title="Scene Reconstruction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(jobs.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
