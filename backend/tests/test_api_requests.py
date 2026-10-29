"""Request handling with a stub model: validation, error schema, CORS, response shape. No weights needed."""

import base64
import io
import struct
import zlib

import numpy as np
from conftest import ALLOWED_ORIGIN, StubEngine, encode, make_client
from PIL import Image

from poultry_disease.preprocess import decode_image, resize_uint8


def post(client, data: bytes, name="x.jpg", ctype="image/jpeg", **params):
    return client.post("/api/v1/predict", files={"file": (name, data, ctype)}, params=params)


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and body["error"]["code"] == code and body["error"]["message"]


def test_health_and_model(stub_client):
    r = stub_client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_loaded"] is True
    info = stub_client.get("/api/v1/model").json()
    assert [c["name"] for c in info["classes"]] == ["Coccidiosis", "Healthy", "Salmonella", "NCD"]
    assert info["run_id"] == "run_20261029_191001"
    assert len(info["weights"]["sha256"]) == 64
    assert 0.9 < info["evaluation"]["internal"]["accuracy"] < 1
    assert info["evaluation"]["external"]["accuracy"] < info["evaluation"]["internal"]["accuracy"]
    assert "Not a veterinary diagnosis" in info["intended_use"]


def test_predict_jpeg_shape_and_verdict(stub_client):
    r = post(stub_client, encode("JPEG", quality=90))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_class"] == "Salmonella"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6
    assert body["verdict"]["category"] == "disease_pattern"
    assert body["verdict"]["runner_up"] == "Coccidiosis"
    assert body["image"] == {"format": "jpeg", "width": 64, "height": 48}
    assert body["cam"] is None
    assert "Not a veterinary diagnosis" in body["disclaimer"]


def test_low_margin_is_uncertain():
    with make_client(StubEngine((0.45, 0.05, 0.45, 0.05))) as client:
        v = post(client, encode("PNG"), "x.png", "image/png").json()["verdict"]
    assert v["category"] == "uncertain"
    assert v["reasons"] == ["low_top_probability", "small_margin"]
    assert v["top_class"] == "Coccidiosis"  # ties keep model order


def test_ncd_is_flagged_notifiable():
    with make_client(StubEngine((0.0, 0.0, 0.0, 1.0))) as client:
        v = post(client, encode("JPEG")).json()["verdict"]
    assert v["notifiable"] is True and v["category"] == "disease_pattern"


def test_cam_payload(stub_client):
    data = encode("JPEG", size=(300, 200))
    body = post(stub_client, data, cam="true").json()
    cam = body["cam"]
    assert cam["target_class"] == "Salmonella"
    for m in cam["maps"].values():
        a = np.asarray(m)
        assert a.shape == (7, 7) and a.min() >= 0 and a.max() <= 1
    seen = np.asarray(Image.open(io.BytesIO(base64.b64decode(cam["input_png_base64"]))))
    np.testing.assert_array_equal(seen, resize_uint8(decode_image(data)))
    overlay = Image.open(io.BytesIO(base64.b64decode(cam["overlay_png_base64"])))
    assert overlay.size == (224, 224) and overlay.mode == "RGB"


def test_png_modes_accepted(stub_client):
    for mode in ("RGBA", "L"):
        assert post(stub_client, encode("PNG", mode=mode), "x.png", "image/png").status_code == 200


def test_heic_accepted(stub_client):
    r = post(stub_client, encode("HEIF", quality=90), "x.heic", "image/heic")
    assert r.status_code == 200, r.text
    assert r.json()["image"]["format"] == "heic"


def test_type_is_sniffed_not_trusted(stub_client):
    assert_error(post(stub_client, b"GIF89a" + b"\0" * 100, "x.jpg", "image/jpeg"), 415, "unsupported_media_type")
    assert_error(post(stub_client, b"%PDF-1.7 hello", "x.png", "image/png"), 415, "unsupported_media_type")
    # a real JPEG with a misleading name and content type is still accepted
    assert post(stub_client, encode("JPEG"), "x.txt", "text/plain").status_code == 200


def test_empty_and_missing_file(stub_client):
    assert_error(post(stub_client, b""), 400, "empty_file")
    assert_error(stub_client.post("/api/v1/predict"), 422, "invalid_request")


def test_corrupt_and_truncated(stub_client):
    data = encode("JPEG", size=(256, 256))
    assert_error(post(stub_client, data[:20]), 400, "invalid_image")
    assert_error(post(stub_client, b"\xff\xd8\xff\xe0" + b"\x00" * 64), 400, "invalid_image")
    png = encode("PNG", size=(256, 256))
    assert_error(post(stub_client, png[: len(png) // 2], "x.png", "image/png"), 400, "invalid_image")


def _png_header_only(width: int, height: int) -> bytes:
    """A tiny PNG whose header claims a huge image (decompression bomb)."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0" * 64)) + chunk(b"IEND", b"")


def test_decompression_bomb_rejected_before_decoding():
    engine = StubEngine()
    with make_client(engine) as client:
        for w, h in ((8_000, 6_000), (50_000, 50_000)):  # over our limit; also over Pillow's own
            bomb = _png_header_only(w, h)
            assert len(bomb) < 200
            assert_error(post(client, bomb, "x.png", "image/png"), 413, "image_too_large")
    assert engine.calls == 0


def test_upload_size_limit():
    with make_client(max_upload_bytes=50_000) as client:
        big = b"\xff\xd8\xff" + b"\0" * 200_000
        assert_error(post(client, big), 413, "file_too_large")
        assert post(client, encode("JPEG")).status_code == 200


def test_upload_size_limit_when_streamed_without_length():
    boundary = "----limit"

    def body():
        yield (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        for _ in range(40):
            yield b"\xff" * 8192
        yield f"\r\n--{boundary}--\r\n".encode()

    with make_client(max_upload_bytes=50_000) as client:
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        r = client.post("/api/v1/predict", content=body(), headers=headers)
        assert r.request.headers.get("transfer-encoding") == "chunked"
        assert_error(r, 413, "file_too_large")


def test_cors_allows_only_configured_origins(stub_client):
    ok = stub_client.get("/api/v1/health", headers={"Origin": ALLOWED_ORIGIN})
    assert ok.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
    bad = stub_client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers
    pre = stub_client.options(
        "/api/v1/predict", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert pre.status_code == 400


def test_unknown_route_uses_error_schema(stub_client):
    assert_error(stub_client.get("/api/v1/nope"), 404, "not_found")


def test_openapi_docs_enabled(stub_client):
    assert stub_client.get("/docs").status_code == 200
    assert "/api/v1/predict" in stub_client.get("/openapi.json").json()["paths"]
