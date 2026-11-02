"""Exercise the original HTTP contract against the real ONNX service/model.

Run from any directory: python 1-Onnx/tests/check_api.py --provider cuda
All saved result images and the temporary test configuration stay in a dedicated
system temporary directory. The project's image directories are not touched.
"""

import argparse
import copy
import importlib
import io
import json
import os
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from fastapi.testclient import TestClient
from PIL import Image


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"image", "discri", "box_thre", "text_thre"}
FOOD_FIELDS = {"image", "dino_cap", "dino_box", "dino_text", "clip_cap", "clip_thre"}
RESULT_FIELDS = {"boxes", "phrases", "logits"}
FORM = {"discri": "glass . straw . lemon .", "box_thre": "0.35", "text_thre": "0.25"}


def _resolve_reference(document, schema):
    while "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/"):
            raise AssertionError(f"Unexpected external OpenAPI reference: {reference}")
        schema = document
        for component in reference[2:].split("/"):
            schema = schema[component.replace("~1", "/").replace("~0", "~")]
    return schema


def _assert_response(response, code, expected=None):
    assert response.status_code == code, (response.status_code, response.text)
    result = response.json()
    if expected is not None:
        assert result == expected, result
    return result


def _check(report, name, action):
    record = {"name": name, "passed": False}
    try:
        details = action()
        if details is not None:
            record["details"] = details
        record["passed"] = True
        print(f"PASS {name}", flush=True)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAIL {name}: {record['error']}", flush=True)
    report["tests"].append(record)


def _assert_no_torch():
    unexpected = sorted(name for name in sys.modules
                        if name == "torch" or name.startswith("torch.")
                        or name == "groundingdino" or name.startswith("groundingdino."))
    assert not unexpected, unexpected[:10]
    return {"torch_imported": False, "groundingdino_imported": False}


def run(args, report):
    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    assert isinstance(settings, dict), "Configuration must be a mapping"
    settings = copy.deepcopy(settings)
    model_settings = settings.setdefault("GROUNDINGDINO_CONFIG", {})
    for key, default in (("model_path", "config/model/groundingdino/groundingdino_swint_ogc.onnx"),
                         ("tokenizer_path", "config/model/groundingdino/bert-base-uncased")):
        path = Path(model_settings.get(key, default))
        model_settings[key] = str((path if path.is_absolute() else BUNDLE_ROOT / path).resolve())
    model_settings["use_gpu"] = args.provider == "cuda"
    settings.setdefault("CLIP_CONFIG", {})["use_gpu"] = args.provider == "cuda"
    image_path = Path(args.image).resolve()
    image_bytes = image_path.read_bytes()
    prior_environment = {name: os.environ.get(name)
                         for name in ("GROUNDING_ONNX_CONFIG", "USE_TORCH", "USE_TF")}
    inserted_path = str(BUNDLE_ROOT)
    service = None
    sys.path.insert(0, inserted_path)
    try:
        with tempfile.TemporaryDirectory(prefix="grounding_onnx_api_contract_") as directory:
            temporary_root = Path(directory).resolve()
            assert temporary_root.parent == Path(tempfile.gettempdir()).resolve()
            config_path = temporary_root / "config.yaml"
            config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
            os.environ["GROUNDING_ONNX_CONFIG"] = str(config_path)
            service = importlib.import_module("main")
            assert Path(service.__file__).resolve() == BUNDLE_ROOT / "main.py", service.__file__
            from app.bootstrap.application import create_app
            from app.core.config.settings import Settings
            app_settings = Settings()
            app_settings = replace(
                app_settings,
                config=replace(app_settings.config, yaml_path=str(config_path)),
                paths=replace(
                    app_settings.paths,
                    save_images_dir=str(temporary_root / "images"),
                    save_result_dir=str(temporary_root / "save_res"),
                    output_dir=str(BUNDLE_ROOT / "output/test_runs" / temporary_root.name),
                ),
                nacos=replace(app_settings.nacos, enabled=False),
                warmup=replace(app_settings.warmup, enabled=False),
            )
            service.app = create_app(app_settings)
            with TestClient(service.app) as client:
                predictor = service.app.state.runtime.predictor
                providers = predictor.session.get_providers()
                expected_provider = "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
                assert providers[0] == expected_provider, providers
                report["active_providers"] = providers
                report["temporary_uploads_isolated"] = True

                def post(data=None, files=None):
                    return client.post("/process/", data=FORM if data is None else data,
                                       files={"image": (image_path.name, image_bytes, "image/jpeg")}
                                       if files is None else files)

                def schema_test():
                    document = _assert_response(client.get("/openapi.json"), 200)
                    operation = document["paths"]["/process/"]["post"]
                    schema = _resolve_reference(
                        document, operation["requestBody"]["content"]["multipart/form-data"]["schema"],
                    )
                    assert set(schema["properties"]) == FIELDS | {"save_result"}, schema
                    assert set(schema["required"]) == FIELDS, schema
                    assert schema["properties"]["save_result"]["type"] == "boolean", schema
                    assert schema["properties"]["save_result"]["default"] is False, schema
                    image_schema = schema["properties"]["image"]
                    assert image_schema["type"] == "string"
                    # FastAPI/Pydantic may represent an uploaded file with the
                    # OpenAPI 3.0 binary format or 3.1 contentMediaType.
                    assert (image_schema.get("format") == "binary" or
                            image_schema.get("contentMediaType") == "application/octet-stream"), image_schema
                    assert schema["properties"]["discri"]["type"] == "string"
                    assert schema["properties"]["box_thre"]["type"] == "number"
                    assert schema["properties"]["text_thre"]["type"] == "number"
                    return {"required_multipart_fields": sorted(FIELDS)}

                _check(report, "openapi_original_fields", schema_test)
                def food_schema_test():
                    document = service.app.openapi()
                    operation = document["paths"]["/food_detect/"]["post"]
                    schema = _resolve_reference(
                        document, operation["requestBody"]["content"]["multipart/form-data"]["schema"],
                    )
                    assert set(schema["properties"]) == FOOD_FIELDS | {"save_result"}, schema
                    assert set(schema["required"]) == FOOD_FIELDS, schema
                    assert schema["properties"]["save_result"]["type"] == "boolean", schema
                    assert schema["properties"]["save_result"]["default"] is False, schema
                    for name in ("dino_cap", "clip_cap"):
                        assert schema["properties"][name]["type"] == "string", schema
                    for name in ("dino_box", "dino_text", "clip_thre"):
                        assert schema["properties"][name]["type"] == "number", schema
                    image_schema = schema["properties"]["image"]
                    assert image_schema["type"] == "string", image_schema
                    assert (image_schema.get("format") == "binary" or
                            image_schema.get("contentMediaType") == "application/octet-stream"), image_schema
                    return {"required_multipart_fields": sorted(FOOD_FIELDS)}
                _check(report, "openapi_food_detect_fields", food_schema_test)

                def previous_schema_test():
                    previous = BUNDLE_ROOT / "tests/fixtures/openapi.json"
                    assert previous.is_file(), f"OpenAPI fixture missing: {previous}"
                    old = json.loads(previous.read_text(encoding="utf-8"))
                    new = service.app.openapi()
                    old_operation = old["paths"]["/process/"]["post"]
                    new_operation = new["paths"]["/process/"]["post"]
                    assert old_operation["responses"] == new_operation["responses"], \
                        "Original /process/ response contract changed"
                    old_body = old_operation["requestBody"]
                    new_body = new_operation["requestBody"]
                    assert old_body["required"] == new_body["required"]
                    assert set(old_body["content"]) == set(new_body["content"])
                    old_fields = _resolve_reference(old, old_body["content"]["multipart/form-data"]["schema"])
                    new_fields = _resolve_reference(new, new_body["content"]["multipart/form-data"]["schema"])
                    assert set(old_fields["required"]) == set(new_fields["required"]), \
                        "Original required fields changed"
                    for name, schema in old_fields["properties"].items():
                        assert new_fields["properties"][name] == schema, f"Original field changed: {name}"
                    assert set(new_fields["properties"]) == set(old_fields["properties"]) | {"save_result"}
                    old_schemas = old.get("components", {}).get("schemas", {})
                    new_schemas = new.get("components", {}).get("schemas", {})
                    for name, schema in old_schemas.items():
                        if schema is old_fields:
                            continue  # The form adds only the optional field checked above.
                        assert new_schemas.get(name) == schema, f"Original schema changed: {name}"
                    return {"original_required_fields_preserved": sorted(old_fields["required"]),
                            "original_response_contract_preserved": True,
                            "optional_save_result_added": True}
                _check(report, "openapi_original_contract_preserved", previous_schema_test)
                _check(report, "no_pytorch_import_at_startup", _assert_no_torch)

                def success_and_roundtrip():
                    upload_directory = service.app.state.runtime.upload_dir
                    before = set(temporary_root.rglob("*.jpg"))
                    result = _assert_response(post(), 200)
                    assert set(result) == RESULT_FIELDS, result.keys()
                    count = len(result["boxes"])
                    assert count > 0, "The real positive image must yield detections"
                    assert count == len(result["phrases"]) == len(result["logits"])
                    assert set(temporary_root.rglob("*.jpg")) == before, \
                        "A request without save_result must not save images"
                    assert not upload_directory.exists(), "Uploaded images must not be written to disk"
                    expected_jpeg = io.BytesIO()
                    with Image.open(io.BytesIO(image_bytes)) as source:
                        if source.mode in ("RGBA", "P", "LA", "PA"):
                            source = source.convert("RGB")
                        source.save(expected_jpeg, format="JPEG", quality=95)
                    expected_jpeg.seek(0)
                    with Image.open(expected_jpeg) as source:
                        decoded_image = source.convert("RGB")
                    try:
                        reference = predictor.predict(decoded_image, FORM["discri"], 0.35, 0.25)
                    finally:
                        decoded_image.close()
                    assert result["phrases"] == reference["phrases"]
                    np.testing.assert_allclose(result["boxes"], reference["boxes"], atol=1e-6, rtol=1e-6)
                    np.testing.assert_allclose(result["logits"], reference["logits"], atol=1e-6, rtol=1e-6)
                    return {"positive_detections": count, "response_fields": sorted(result),
                            "no_images_saved_by_default": True,
                            "http_jpeg_quality_95_prediction_tolerance": 1e-6}

                _check(report, "positive_image_and_in_memory_jpeg_prediction", success_and_roundtrip)

                def save_result_test():
                    result_directory = service.app.state.runtime.result_dir
                    saved_paths = []
                    with Image.open(io.BytesIO(image_bytes)) as source:
                        expected_size = source.size
                    for _ in range(2):
                        before = set(result_directory.glob("*.jpg"))
                        result = _assert_response(post(data={**FORM, "save_result": "true"}), 200)
                        assert set(result) == RESULT_FIELDS, result.keys()
                        saved = set(result_directory.glob("*.jpg")) - before
                        assert len(saved) == 1, saved
                        destination = saved.pop()
                        with Image.open(destination) as image:
                            assert image.format == "JPEG"
                            assert image.size == expected_size
                            image.load()
                        saved_paths.append(destination)
                    assert len(set(saved_paths)) == 2, "Saved results must have unique names"
                    assert not service.app.state.runtime.upload_dir.exists()
                    before = set(temporary_root.rglob("*.jpg"))
                    _assert_response(post(data={**FORM, "save_result": "false"}), 200)
                    assert set(temporary_root.rglob("*.jpg")) == before
                    return {"saved_results": len(saved_paths), "unique_names": True,
                            "explicit_false_saves_no_images": True}

                _check(report, "optional_save_result_images", save_result_test)
                _check(report, "invalid_save_result_422", lambda: _assert_response(
                    post(data={**FORM, "save_result": "not-a-bool"}), 422,
                ))

                def rgba_test():
                    with Image.open(io.BytesIO(image_bytes)) as source:
                        rgba = source.convert("RGBA")
                    rgba.putalpha(128)
                    buffer = io.BytesIO()
                    rgba.save(buffer, format="PNG")
                    result = _assert_response(post(files={"image": ("rgba.png", buffer.getvalue(), "image/png")}), 200)
                    assert set(result) == RESULT_FIELDS
                    assert len(result["boxes"]) == len(result["phrases"]) == len(result["logits"])
                    return {"detections": len(result["boxes"])}

                _check(report, "rgba_upload", rgba_test)
                _check(report, "non_image_mime_400", lambda: _assert_response(
                    post(files={"image": ("test.txt", image_bytes, "text/plain")}), 400,
                    {"error": "Uploaded file is not an image"},
                ))
                _check(report, "corrupt_image_400", lambda: _assert_response(
                    post(files={"image": ("broken.jpg", b"not an encoded image", "image/jpeg")}), 400,
                    {"error": "Invalid image format or corrupted image"},
                ))
                for field in sorted(FIELDS):
                    def missing_test(field=field):
                        data = {name: value for name, value in FORM.items() if name != field}
                        result = _assert_response(post(data=data, files={} if field == "image" else None), 422)
                        assert any(error["loc"][-1] == field for error in result["detail"]), result
                        return {"missing": field}
                    _check(report, f"missing_required_{field}_422", missing_test)
                for field in ("box_thre", "text_thre"):
                    _check(report, f"invalid_float_{field}_422", lambda field=field: _assert_response(
                        post(data={**FORM, field: "not-a-float"}), 422,
                    ))
                _check(report, "out_of_range_threshold_original_behavior", lambda: _assert_response(
                    post(data={**FORM, "box_thre": "1.1"}), 200,
                    {"boxes": [], "phrases": [], "logits": []},
                ))

                def failure_test():
                    original_predict = predictor.predict
                    def fail_prediction(*unused_args, **unused_kwargs):
                        raise RuntimeError("intentional contract test error")
                    try:
                        predictor.predict = fail_prediction
                        return _assert_response(post(), 500, {
                            "error": "Model inference failed", "details": "intentional contract test error",
                        })
                    finally:
                        predictor.predict = original_predict
                _check(report, "inference_error_500", failure_test)
                _check(report, "no_pytorch_import_after_inference", _assert_no_torch)
                def log_test():
                    from app.core.config.settings import local_path
                    log_dir = local_path(app_settings.paths.output_dir) / "logs"
                    records = [json.loads(line) for line in (log_dir / "platform.log").read_text(encoding="utf-8").splitlines()]
                    app_log = (log_dir / "app.log").read_text(encoding="utf-8")
                    # 不输出 HTTP 请求摘要：普通日志和平台日志中都没有 "HTTP 请求完成"。
                    assert not any(record["message"] == "HTTP 请求完成" for record in records), \
                        "HTTP request summaries must not reach platform.log"
                    assert "HTTP 请求完成" not in app_log, "HTTP request summaries must not reach app.log"
                    assert any(record["stack_trace"] and "intentional contract test error" in record["stack_trace"]
                               and record.get("traceId") for record in records), "inference failure without trace"
                    assert any(record["message"] == "推理完成" and "inference_ms" in record and record["traceId"]
                               for record in records), "no platform 推理完成 record with inference_ms"
                    return {"json_records": len(records), "http_request_summaries": 0, "trace_id_and_timings": True}
                _check(report, "application_and_platform_logs", log_test)
    finally:
        for name, value in prior_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if sys.path and sys.path[0] == inserted_path:
            sys.path.pop(0)
    report["passed"] = bool(report["tests"] and all(test["passed"] for test in report["tests"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--config", default=str(BUNDLE_ROOT / "config/param/config.yaml"))
    parser.add_argument("--image", default=str(BUNDLE_ROOT / "config/images/test.jpg"))
    parser.add_argument("--report", help="Default: reports/api_contract.<provider>.json")
    args = parser.parse_args()
    report = {"passed": False, "provider": args.provider, "tests": [], "failures": []}
    try:
        run(args, report)
    except Exception as exc:
        report["passed"] = False
        report["failures"].append(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    target = Path(args.report) if args.report else BUNDLE_ROOT / f"reports/api_contract.{args.provider}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"API contract {'PASSED' if report['passed'] else 'FAILED'}; report: {target}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
