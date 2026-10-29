"""FastAPI app: ``GET /api/v1/health``, ``GET /api/v1/model``, ``POST /api/v1/predict[?cam=true]``.

Run locally:  uvicorn poultry_api.main:app --app-dir backend --port 8000   (OpenAPI docs at /docs)
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from poultry_api import API_PREFIX
from poultry_api.engine import Engine, KerasEngine
from poultry_api.errors import ApiError, error_response, install_handlers
from poultry_api.imaging import load_image
from poultry_api.model_info import load_model_info
from poultry_api.schemas import ErrorResponse, Health, PredictResponse
from poultry_api.settings import Settings
from poultry_disease import CLASS_NAMES, __version__
from poultry_disease.cam import normalize_cam, overlay, to_png
from poultry_disease.verdict import interpret

ERRORS = {code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500)}
#: Room for the multipart boundary and headers on top of the file itself.
MULTIPART_OVERHEAD = 64 * 1024


class BodySizeLimit:
    """Reject request bodies over the limit while they stream in, before multipart parsing spools them."""

    def __init__(self, app: ASGIApp, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        length = headers.get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.limit:
            return await self._reject(scope, receive, send)

        seen = 0

        async def limited() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise ApiError(413, "file_too_large", self._message())
            return message

        try:
            await self.app(scope, limited, send)
        except ApiError as exc:  # raised while streaming, before a response was started
            await error_response(exc.status, exc.code, exc.message)(scope, receive, send)

    def _message(self) -> str:
        return f"The upload is larger than {(self.limit - MULTIPART_OVERHEAD) / 1024 / 1024:.0f} MB."

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        await error_response(413, "file_too_large", self._message())(scope, receive, send)


def create_app(settings: Settings | None = None, engine_factory: Callable[[Settings], Engine] | None = None):
    settings = settings or Settings()
    engine_factory = engine_factory or (lambda s: KerasEngine(allow_download=s.allow_download))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_info = load_model_info(settings.repo_root)
        app.state.engine = engine_factory(settings)  # loaded once; requests never reload it
        yield

    app = FastAPI(
        title="Poultry disease image classifier API",
        version=__version__,
        description=(
            "Classifies a photo of poultry droppings into Coccidiosis, Healthy, Salmonella or Newcastle disease "
            "(NCD) with a fine-tuned ResNet50. Research prototype: **not a veterinary diagnosis**. About 97% "
            "accuracy on the internal test set but about 80% on PCR-confirmed images from other farms; see "
            "`GET /api/v1/model`."
        ),
        lifespan=lifespan,
    )
    install_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )
    app.add_middleware(BodySizeLimit, limit=settings.max_upload_bytes + MULTIPART_OVERHEAD)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get(f"{API_PREFIX}/health", response_model=Health, tags=["service"])
    def health(request: Request):
        return Health(status="ok", model_loaded=request.app.state.engine is not None, model_version=__version__)

    @app.get(f"{API_PREFIX}/model", tags=["service"], summary="Model card summary, read from the committed files")
    def model(request: Request) -> dict:
        return request.app.state.model_info

    @app.post(f"{API_PREFIX}/predict", response_model=PredictResponse, responses=ERRORS, tags=["inference"])
    async def predict(
        request: Request,
        file: UploadFile = File(description="JPEG, PNG or HEIC photo of poultry droppings"),  # noqa: B008
        cam: bool = Query(False, description="Also return the class activation maps and a heatmap overlay"),
    ):
        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise ApiError(413, "file_too_large", f"The upload is larger than {settings.max_upload_bytes >> 20} MB.")
        return await run_in_threadpool(_predict, request.app.state.engine, data, cam, settings.max_pixels)

    return app


def _png_b64(rgb: np.ndarray) -> str:
    return base64.b64encode(to_png(rgb)).decode("ascii")


def _predict(engine: Engine, data: bytes, with_cam: bool, max_pixels: int) -> PredictResponse:
    image = load_image(data, max_pixels)
    probs, cam = engine.predict(image.seen)
    probs = np.asarray(probs, dtype=np.float64)
    v = interpret(probs.tolist())
    cam_payload = None
    if with_cam:
        maps = {c: normalize_cam(cam[..., i]) for i, c in enumerate(CLASS_NAMES)}
        cam_payload = {
            "target_class": v.top_class,
            "maps": {c: np.round(m, 6).tolist() for c, m in maps.items()},
            "overlay_png_base64": _png_b64(overlay(image.seen, maps[v.top_class])),
            "input_png_base64": _png_b64(image.seen),
        }
    return PredictResponse(
        model_version=__version__,
        predicted_class=v.top_class,
        confidence=v.top_probability,
        probabilities={c: float(p) for c, p in zip(CLASS_NAMES, probs, strict=True)},
        low_confidence=v.low_confidence,
        verdict={
            "category": v.category,
            "top_class": v.top_class,
            "runner_up": v.runner_up,
            "top_probability": v.top_probability,
            "margin": v.margin,
            "reasons": list(v.reasons),
            "notifiable": v.notifiable,
        },
        image={"format": image.format, "width": image.width, "height": image.height},
        cam=cam_payload,
    )


app = create_app()
