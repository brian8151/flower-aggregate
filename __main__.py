import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.router.agent_router import agent_router
import logging
import sys
from os.path import dirname as opd, realpath as opr

from src.util.app_util import print_tensorflow_version

_BASEDIR_ = opd(opr(__file__))
sys.path.append(_BASEDIR_)
from dotenv import load_dotenv
load_dotenv(dotenv_path=_BASEDIR_+"/app.env")

def create_app():
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    app = FastAPI()

    @app.on_event("startup")
    def on_startup():
        try:
            logging.info("Aikya FL Client started and is listening on http://0.0.0.0:7000")
            print_tensorflow_version()
        except Exception as e:
            logging.error("Unexpected error: %s", e)

    @app.on_event("shutdown")
    def on_shutdown():
        pass

    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    app.include_router(agent_router, prefix="/agent/api")
    return app


# Legacy mode
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=7000)
