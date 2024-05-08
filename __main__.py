import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.router.flower_router import flower_router
import logging
def create_app():
    app = FastAPI()

    @app.on_event("startup")
    def on_startup():
        logging.info("Server started and is listening on http://0.0.0.0:8000")
    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    app.include_router(flower_router, prefix="/flwr")
    return app


# Legacy mode
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)