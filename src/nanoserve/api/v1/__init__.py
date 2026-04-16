from fastapi.routing import APIRouter
from .chat import router as chat_router
from .models import router as models_router

router = APIRouter(prefix="/v1", tags=["v1"])
router.include_router(chat_router)
router.include_router(models_router)