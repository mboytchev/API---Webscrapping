import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.app import get_application
from src.api.routes.data import router as data_router


app = get_application()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

app.include_router(data_router, prefix="/data", tags=["data"])


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, port=8080)

