from fastapi import APIRouter

from app.api.v1.endpoints import auth, dashboard, predict, traffic

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(predict.router, tags=["prediction"])
api_router.include_router(traffic.router, prefix="/traffic", tags=["traffic"])

# Page routes (HTML, not JSON) are mounted without the /api/v1 JSON prefix
# in main.py — kept here just for discoverability/grouping.
pages_router = dashboard.router
