import uvicorn
from server.server import app

def main():
    """Entry point for the OpenEnv multi-mode deployment."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()