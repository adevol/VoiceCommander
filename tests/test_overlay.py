import queue
import unittest

from voicecommander.preview import PreviewOverlay


class _Root:
    def __init__(self):
        self.withdrawn = 0
        self.shown = 0
        self.lifted = 0
        self.geometries = []

    def withdraw(self):
        self.withdrawn += 1

    def update_idletasks(self):
        pass

    def winfo_screenwidth(self):
        return 1000

    def geometry(self, value):
        self.geometries.append(value)

    def deiconify(self):
        self.shown += 1

    def lift(self):
        self.lifted += 1


class _Text:
    def __init__(self):
        self.value = None

    def configure(self, **kwargs):
        self.value = kwargs["text"]

    def winfo_reqwidth(self):
        return 100

    def winfo_reqheight(self):
        return 20


def _overlay():
    overlay = PreviewOverlay.__new__(PreviewOverlay)
    overlay.root = _Root()
    overlay.text = _Text()
    return overlay


class PreviewOverlayTests(unittest.TestCase):
    def test_pump_renders_only_latest_text(self):
        overlay = _overlay()
        updates = queue.SimpleQueue()
        updates.put("old")
        updates.put("latest")

        overlay.pump(updates)

        self.assertEqual(overlay.text.value, "latest")
        self.assertEqual(overlay.root.shown, 1)
        self.assertEqual(overlay.root.lifted, 1)

    def test_pump_latest_none_hides_overlay(self):
        overlay = _overlay()
        updates = queue.SimpleQueue()
        updates.put("old")
        updates.put(None)

        overlay.pump(updates)

        self.assertEqual(overlay.root.withdrawn, 1)
        self.assertEqual(overlay.root.shown, 0)

    def test_pump_empty_queue_leaves_overlay_unchanged(self):
        overlay = _overlay()

        overlay.pump(queue.SimpleQueue())

        self.assertEqual(overlay.root.withdrawn, 0)
        self.assertEqual(overlay.root.shown, 0)
        self.assertIsNone(overlay.text.value)

    def test_pump_can_show_text_after_hide(self):
        overlay = _overlay()
        updates = queue.SimpleQueue()
        updates.put(None)
        overlay.pump(updates)

        updates.put("new text")
        overlay.pump(updates)

        self.assertEqual(overlay.root.withdrawn, 1)
        self.assertEqual(overlay.root.shown, 1)
        self.assertEqual(overlay.text.value, "new text")


if __name__ == "__main__":
    unittest.main()
