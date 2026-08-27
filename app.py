import uvicorn
from src.config import DEFAULT_PORT
from src.api.routes import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT)
