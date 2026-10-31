"""Check combined ChineseCLIP inference without loading model weights."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from app.domain.pipeline.b_clipdtct import ClipDetect


class ClipFoodFilterChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.asset_dir = Path(temporary.name)
        self.model_path = self.asset_dir / "model.onnx"
        self.model_path.touch()
        self.model_config = {
            "model_type": "chinese_clip",
            "projection_dim": 2,
            "text_config": {"max_position_embeddings": 8},
            "vision_config": {"image_size": 4},
        }
        self.preprocessor = {
            "do_resize": True,
            "size": {"height": 4, "width": 6},
            "resample": 3,
            "do_center_crop": False,
            # A disabled center crop must not override the declared resize size.
            "crop_size": {"height": 2, "width": 2},
            "do_convert_rgb": True,
            "do_rescale": True,
            "rescale_factor": 1 / 255,
            "do_normalize": True,
            "image_mean": [0.1, 0.2, 0.3],
            "image_std": [0.5, 0.25, 0.125],
        }
        self.config = {
            "CLIP_CONFIG": {
                "model_path": str(self.model_path),
                "use_gpu": False,
                "threads": 2,
                "threshold": 0.5,
            },
            "TOTAL_CONFIG": {"device_id": 3},
        }
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                 "鱼": 4, "肉": 5, "fish": 6}
        tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
        )
        tokenizer.save(str(self.asset_dir / "tokenizer.json"))
        (self.asset_dir / "tokenizer_config.json").write_text(json.dumps({
            "model_max_length": 1000000000000000019884624838656,
        }), encoding="utf-8")

    def session(self):
        session = Mock()
        session.get_inputs.return_value = [
            SimpleNamespace(name="input_ids", type="tensor(int64)", shape=["texts", "tokens"]),
            SimpleNamespace(name="pixel_values", type="tensor(float)", shape=["images", 3, 4, 6]),
            SimpleNamespace(name="attention_mask", type="tensor(int64)", shape=["texts", "tokens"]),
        ]
        session.get_outputs.return_value = [
            SimpleNamespace(name=name, shape=["batch", 2])
            for name in ("logits_per_image", "logits_per_text", "text_embeds", "image_embeds")
        ]

        def run(names, feed):
            values = {
                "image_embeds": np.tile([3.0, 4.0], (len(feed["pixel_values"]), 1)),
                "text_embeds": np.tile([2.0, 0.0], (len(feed["input_ids"]), 1)),
            }
            return [np.asarray(values[name], dtype=np.float32) for name in names]

        session.run.side_effect = run
        return session

    def detector(self, *, session=None, clip=None):
        (self.asset_dir / "config.json").write_text(json.dumps(self.model_config), encoding="utf-8")
        (self.asset_dir / "preprocessor_config.json").write_text(
            json.dumps(self.preprocessor), encoding="utf-8")
        if clip:
            self.config["CLIP_CONFIG"].update(clip)
        session = self.session() if session is None else session
        with patch("app.domain.pipeline.b_clipdtct.create_session", return_value=session) as create:
            detector = ClipDetect(self.config)
        self.assertTrue(detector.ini_flag)
        return detector, session, create

    def test_initialization_loads_one_combined_session_with_deployment_settings(self):
        detector, session, create = self.detector()
        create.assert_called_once_with(str(self.model_path), "cpu", 2, device_id=3)
        self.assertIs(detector.sess, session)
        self.assertEqual(detector.context_length, 8)
        self.assertEqual(detector.batch_size, 1)
        session.run.assert_not_called()

    def test_constructor_accepts_yaml_path_without_a_separate_text_model(self):
        # The application's YAML loader also validates the GDINO settings.
        self.detector()
        config_path = self.asset_dir / "config.yaml"
        config_path.write_text(
            "GROUNDINGDINO_CONFIG:\n"
            "  model_path: config/model/groundingdino/model.onnx\n"
            "  tokenizer_path: config/model/groundingdino/tokenizer\n"
            "CLIP_CONFIG:\n"
            f"  model_path: {self.model_path.as_posix()}\n"
            "  use_gpu: false\n"
            "  threads: 2\n"
            "  threshold: 0.5\n",
            encoding="utf-8",
        )
        for source in (config_path, str(config_path)):
            with self.subTest(source=type(source).__name__), patch.object(ClipDetect, "init_model"):
                detector = ClipDetect(source)
            self.assertEqual(detector.provider, "cpu")
            self.assertEqual(detector.threads, 2)
            self.assertEqual(detector.threshold, 0.5)
            self.assertEqual(Path(detector.model_path), self.model_path)

    def test_local_tokenizer_supplies_chinese_tokens_special_ids_and_padding_mask(self):
        detector, _, _ = self.detector()
        tokens = detector._tokenize(["鱼", "鱼肉", "FISH"])
        np.testing.assert_array_equal(tokens["input_ids"], [[2, 4, 3, 0], [2, 4, 5, 3], [2, 6, 3, 0]])
        np.testing.assert_array_equal(tokens["attention_mask"], [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]])
        self.assertEqual(tokens["input_ids"].dtype, np.int64)
        self.assertEqual(tokens["attention_mask"].dtype, np.int64)

    def test_context_limit_truncates_text_and_keeps_end_token(self):
        detector, _, _ = self.detector(clip={"context_length": 6})
        tokens = detector._tokenize(["鱼肉鱼肉鱼肉", "鱼"])
        np.testing.assert_array_equal(tokens["input_ids"],
                                      [[2, 4, 5, 4, 5, 3], [2, 4, 3, 0, 0, 0]])
        np.testing.assert_array_equal(tokens["attention_mask"], [[1] * 6, [1, 1, 1, 0, 0, 0]])

    def test_context_defaults_to_512_without_using_tokenizer_sentinel(self):
        self.model_config["text_config"] = {}
        detector, _, _ = self.detector()
        self.assertEqual(detector.context_length, 512)
        self.assertEqual(detector._tokenize(["鱼"])["input_ids"].shape, (1, 3))

    def test_text_inference_requires_nonempty_explicit_prompts(self):
        detector, session, _ = self.detector()
        for texts in (None, [], [""], [" "], [3]):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                detector._tokenize(texts)
        session.run.assert_not_called()

    def test_preprocessing_uses_metadata_rgb_bicubic_resize_without_center_crop(self):
        detector, _, _ = self.detector()
        bgr = np.arange(7 * 19 * 3, dtype=np.uint8).reshape(7, 19, 3)
        tensor = detector.preprocess_image(bgr)
        rgb = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
        expected = np.asarray(rgb.resize((6, 4), Image.Resampling.BICUBIC), dtype=np.float32) / 255
        expected = (expected - np.array(self.preprocessor["image_mean"], dtype=np.float32))
        expected /= np.array(self.preprocessor["image_std"], dtype=np.float32)
        self.assertEqual(tensor.shape, (1, 3, 4, 6))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0], expected.transpose(2, 0, 1), atol=1e-6)

    def test_preprocessing_honors_disabled_rescaling_and_normalization(self):
        self.preprocessor.update(do_rescale=False, do_normalize=False)
        detector, _, _ = self.detector()
        tensor = detector.preprocess_image(np.full((2, 3, 3), [10, 40, 240], dtype=np.uint8))
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [240.0, 40.0, 10.0])

    def test_combined_model_receives_all_three_tensors_and_normalizes_both_outputs(self):
        detector, session, _ = self.detector()
        pixels = np.zeros((2, 3, 4, 6), dtype=np.float32)
        tokens = detector._tokenize(["鱼", "鱼肉"])
        images, texts = detector._run_model(pixels, tokens)
        session.run.assert_called_once()
        names, feed = session.run.call_args.args
        self.assertEqual(names, ["image_embeds", "text_embeds"])
        self.assertEqual(set(feed), {"input_ids", "attention_mask", "pixel_values"})
        self.assertEqual(feed["pixel_values"].dtype, np.float32)
        self.assertEqual(feed["pixel_values"].shape, (2, 3, 4, 6))
        self.assertEqual(feed["input_ids"].dtype, np.int64)
        self.assertEqual(feed["attention_mask"].dtype, np.int64)
        np.testing.assert_allclose(images, [[0.6, 0.8], [0.6, 0.8]], rtol=1e-6)
        np.testing.assert_allclose(texts, [[1.0, 0.0], [1.0, 0.0]], rtol=1e-6)

    def test_single_prompt_returns_mapped_cosine_with_joint_text_inference(self):
        detector, session, _ = self.detector()
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        self.assertAlmostEqual(detector.similarity(image, "鱼"), 0.8, places=6)
        self.assertEqual(session.run.call_count, 1)
        feed = session.run.call_args.args[1]
        np.testing.assert_array_equal(feed["input_ids"], [[2, 4, 3]])
        self.assertEqual(feed["pixel_values"].shape, (1, 3, 4, 6))

    def test_mapping_handles_opposite_orthogonal_aligned_and_roundoff(self):
        detector, _, _ = self.detector()
        for image, expected in (([-1.0, 0.0], 0.0), ([0.0, 1.0], 0.5),
                                ([1.0, 0.0], 1.0), ([1.0000001, 0.0], 1.0),
                                ([-1.0000001, 0.0], 0.0)):
            with self.subTest(image=image):
                self.assertEqual(float(detector._similarity_scores([image], [[1.0, 0.0]])[0]), expected)
        np.testing.assert_allclose(
            detector._similarity_scores([[0.6, 0.8]], [[1.0, 0.0], [0.0, 1.0]]), [0.8, 0.9])

    def test_batch_keeps_crop_coordinates_order_padding_and_strict_threshold(self):
        detector, session, _ = self.detector(clip={"batch_size": 2})
        session.run.side_effect = [
            [np.array([[0.0, 1.0], [3.0, 4.0]]), np.array([[2.0, 0.0]])],
            [np.array([[4.0, 3.0]]), np.array([[2.0, 0.0]])],
        ]
        pixels = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        image = Image.fromarray(pixels)
        boxes = np.array([[0.5, 0.5, 0.4, 0.5], [0.0, 0.5, 0.3, 0.5],
                          [0.8, 0.25, 0.2, 0.5]], dtype=np.float32)
        expected_rects = [[3, 2, 7, 6], [-1, 2, 1, 6], [7, 0, 9, 4]]
        with patch.object(detector, "_tokenize", wraps=detector._tokenize) as tokenize, \
                patch.object(detector, "preprocess_image", wraps=detector.preprocess_image) as preprocess:
            rects, scores = detector.batch_detect(image, boxes, "鱼", 0.5)
        self.assertEqual(rects, expected_rects[1:])
        np.testing.assert_allclose(scores, [0.8, 0.9], rtol=1e-6)
        tokenize.assert_called_once_with(["鱼"])
        self.assertEqual(session.run.call_count, 2)
        self.assertEqual([call.args[1]["pixel_values"].shape[0]
                          for call in session.run.call_args_list], [2, 1])
        for call, rect in zip(preprocess.call_args_list, expected_rects):
            crop = call.args[0]
            self.assertTrue(crop.flags.c_contiguous)
            np.testing.assert_array_equal(crop, np.asarray(image.crop(tuple(rect)))[:, :, ::-1])
        np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_empty_batch_returns_empty_lists_without_tokenization_or_inference(self):
        detector, session, _ = self.detector()
        with patch.object(detector, "_tokenize") as tokenize:
            for boxes in ([], np.empty((0, 4), dtype=np.float32)):
                self.assertEqual(detector.batch_detect(Image.new("RGB", (3, 2)), boxes, "", 0.5), ([], []))
        tokenize.assert_not_called()
        session.run.assert_not_called()

    def test_public_methods_reject_invalid_prompts_thresholds_and_uninitialized_models(self):
        detector, session, _ = self.detector()
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            detector.similarity(image, " ")
        for threshold in (-0.1, 21, np.nan):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                detector.batch_detect(Image.fromarray(image), [[0.5, 0.5, 1, 1]], "鱼", threshold)
        detector.ini_flag = False
        with self.assertRaises(RuntimeError):
            detector.similarity(image, "鱼")
        session.run.assert_not_called()

    def test_invalid_or_mismatched_model_embeddings_raise(self):
        for images, texts in ((np.zeros((1, 2)), [[1, 0]]),
                              ([[np.nan, 1]], [[1, 0]]),
                              ([[1, 0]], np.empty((0, 2))),
                              ([[1, 0]], [[1, 0, 0]])):
            with self.subTest(images=images, texts=texts):
                detector, session, _ = self.detector()
                session.run.side_effect = None
                session.run.return_value = [np.asarray(images), np.asarray(texts)]
                with self.assertRaises(RuntimeError):
                    detector.similarity(np.zeros((2, 2, 3), dtype=np.uint8), "鱼")

    def test_failed_session_initialization_leaves_detector_unavailable(self):
        # Write the metadata first, then exercise the same startup path on failure.
        self.detector()
        with patch("app.domain.pipeline.b_clipdtct.create_session", side_effect=RuntimeError("bad model")), \
                self.assertLogs("clip", level="ERROR"):
            detector = ClipDetect(self.config)
        self.assertFalse(detector.ini_flag)

    def test_incompatible_graph_input_dtype_is_rejected_during_initialization(self):
        self.detector()
        session = self.session()
        session.get_inputs.return_value[0].type = "tensor(float)"
        with patch("app.domain.pipeline.b_clipdtct.create_session", return_value=session), \
                self.assertLogs("clip", level="ERROR"):
            detector = ClipDetect(self.config)
        self.assertFalse(detector.ini_flag)
        session.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
