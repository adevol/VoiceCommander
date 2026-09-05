import queue
import unittest
from unittest.mock import Mock, patch
from sys import modules

from voicecommander.local_asr import PreviewText
from voicecommander.preview import PreviewOverlay


def _overlay():
    overlay = PreviewOverlay.__new__(PreviewOverlay)
    overlay.root = Mock()
    overlay.root.winfo_screenwidth.return_value = 1000
    overlay.text = Mock()
    overlay.text.winfo_reqwidth.return_value = 100
    overlay.text.winfo_reqheight.return_value = 20
    return overlay


class PreviewOverlayTests(unittest.TestCase):
    def test_preview_overlay_consumes_worker_updates(self) -> None:
        tkinter = Mock()
        tkinter.Tk.return_value.winfo_screenwidth.return_value = 1920
        tkinter.Label.return_value.winfo_reqwidth.return_value = 300
        tkinter.Label.return_value.winfo_reqheight.return_value = 40
        updates = queue.SimpleQueue()
        updates.put(PreviewText("Hello", " world", 1.0))

        with patch.dict(modules, {"tkinter": tkinter}):
            overlay = PreviewOverlay()
            overlay.pump(updates)

        tkinter.Label.return_value.configure.assert_called_once_with(text="Hello world")
        tkinter.Tk.return_value.deiconify.assert_called_once()
        tkinter.Tk.return_value.update.assert_not_called()

    def test_pump_renders_only_latest_text(self):
        overlay = _overlay()
        updates = queue.SimpleQueue()
        updates.put("old")
        updates.put("latest")

        overlay.pump(updates)

        overlay.text.configure.assert_called_once_with(text="latest")
        overlay.root.deiconify.assert_called_once_with()
        overlay.root.lift.assert_called_once_with()

    def test_pump_latest_none_hides_overlay(self):
        overlay = _overlay()
        updates = queue.SimpleQueue()
        updates.put("old")
        updates.put(None)

        overlay.pump(updates)

        overlay.root.withdraw.assert_called_once_with()
        overlay.root.deiconify.assert_not_called()

    def test_pump_empty_queue_leaves_overlay_unchanged(self):
        overlay = _overlay()

        overlay.pump(queue.SimpleQueue())

        overlay.root.withdraw.assert_not_called()
        overlay.root.deiconify.assert_not_called()
        overlay.text.configure.assert_not_called()

    def test_pump_can_show_text_after_hide(self):
        overlay = _overlay()
        updates = queue.SimpleQueue()
        updates.put(None)
        overlay.pump(updates)

        updates.put("new text")
        overlay.pump(updates)

        overlay.root.withdraw.assert_called_once_with()
        overlay.root.deiconify.assert_called_once_with()
        overlay.text.configure.assert_called_once_with(text="new text")


if __name__ == "__main__":
    unittest.main()
