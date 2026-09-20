"""Regression checks for user-provided manifests; no checkpoints or GPU required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("freezeempath_infer", ROOT / "inference/infer.py")
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "example.wav"
        self.audio.write_bytes(b"fixture")
        self.manifest = self.root / "manifest.jsonl"

    def write(self, text):
        self.manifest.write_text(text, encoding="utf-8")

    def test_audio_only_and_manifest_relative_path(self):
        self.write('\n' + json.dumps({"audio": self.audio.name}) + '\n')
        rows = infer.read_questions(self.manifest)
        self.assertEqual(rows[0][1], self.audio.resolve())
        self.assertNotIn("emotion", rows[0][0])

    def test_absolute_path(self):
        self.write(json.dumps({"id": "中文", "audio": str(self.audio)}))
        self.assertEqual(infer.read_questions(self.manifest)[0][0]["id"], "中文")

    def test_bad_inputs_fail_before_model_loading(self):
        for text, error in [
            ('', ValueError), ('\n', ValueError), ('{broken}', ValueError),
            ('[]', ValueError), ('{"text":"hello"}', ValueError),
            ('{"audio":"missing.wav"}', FileNotFoundError),
        ]:
            with self.subTest(text=text):
                self.write(text)
                with self.assertRaises(error):
                    infer.read_questions(self.manifest)

    def test_existing_examples(self):
        # Existing manifests use paths relative to the repository root.
        import os
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            rows = infer.read_questions(ROOT / "examples/manifest.jsonl")
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(audio.is_file() for _, audio in rows))
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
