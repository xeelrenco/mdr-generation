from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mdr_generator.sow_paths import file_sha256, sow_content_hashes, sow_fingerprint


class SowFingerprintTests(unittest.TestCase):
    def _pdf(self, tmp: str, name: str, content: bytes) -> Path:
        path = Path(tmp) / name
        path.write_bytes(content)
        return path

    def test_same_content_same_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = self._pdf(tmp, "sow.pdf", b"%PDF content A")
            self.assertEqual(sow_fingerprint([a]), sow_fingerprint([a]))

    def test_changed_content_changes_fingerprint(self) -> None:
        """Stesso nome file, contenuto diverso: impronta diversa."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._pdf(tmp, "sow.pdf", b"%PDF content A")
            before = sow_fingerprint([path])
            path.write_bytes(b"%PDF content B")
            self.assertNotEqual(before, sow_fingerprint([path]))

    def test_added_pdf_changes_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = self._pdf(tmp, "a.pdf", b"A")
            b = self._pdf(tmp, "b.pdf", b"B")
            self.assertNotEqual(sow_fingerprint([a]), sow_fingerprint([a, b]))

    def test_pdf_order_does_not_matter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = self._pdf(tmp, "a.pdf", b"A")
            b = self._pdf(tmp, "b.pdf", b"B")
            self.assertEqual(sow_fingerprint([a, b]), sow_fingerprint([b, a]))

    def test_unreadable_file_never_matches_a_real_hash(self) -> None:
        missing = Path("does-not-exist.pdf")
        digest = file_sha256(missing)
        self.assertTrue(digest.startswith("unreadable:"))
        with tempfile.TemporaryDirectory() as tmp:
            real = self._pdf(tmp, "sow.pdf", b"%PDF")
            self.assertNotEqual(digest, file_sha256(real))

    def test_hashes_can_be_keyed_by_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = self._pdf(tmp, "sow.pdf", b"A")
            hashes = sow_content_hashes([a], {a: "sow.pdf#deadbeef"})
            self.assertEqual(list(hashes), ["sow.pdf#deadbeef"])


if __name__ == "__main__":
    unittest.main()
