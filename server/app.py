# server/app.py — required by openenv validate for multi-mode deployment detection
# Re-exports the FastAPI app from the root server.py so openenv can locate it.
from server import app

__all__ = ["app"]