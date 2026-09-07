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
    overlay.text.winfo_width.return_value = 100
    overlay.text.winfo_reqheight.return_value = 20
    overlay.text.count.return_value = 0
    return overlay


class PreviewOverlayTests(unittest.TestCase):
    def test_preview_overlay_consumes_worker_updates(self) -> None:
        tkinter = Mock()
        tkinter.Tk.return_value.winfo_screenwidth.return_value = 1920
        tkinter.Text.return_value.winfo_reqwidth.return_value = 300
        tkinter.Text.return_value.winfo_width.return_value = 300
        tkinter.Text.return_value.winfo_reqheight.return_value = 40
        tkinter.Text.return_value.count.return_value = 0
        updates = queue.SimpleQueue()
        updates.put(PreviewText("Hello", " world", 1.0))

        with patch.dict(modules, {"tkinter": tkinter}):
            overlay = PreviewOverlay()
            overlay.pump(updates)

        text = tkinter.Text.return_value
        self.assertEqual(
            [call.args for call in text.insert.call_args_list],
            [("end", "Hello"), ("end", " world", "tentative")],
        )
        text.tag_configure.assert_called_once_with("tentative", foreground="#a1a1aa")
        text.configure.assert_any_call(state="disabled")
        tkinter.Tk.return_value.deiconify.assert_called_once()
        tkinter.Tk.return_value.update.assert_not_called()

    def test_pump_renders_only_latest_text(self):
        overlay = _overlay()
        updates = queue.SimpleQueue()
        updates.put("old")
        updates.put("latest")

        overlay.pump(updates)

        overlay.text.insert.assert_called_once_with("end", "latest")
        overlay.text.delete.assert_called_once_with("1.0", "end")
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
        overlay.text.insert.assert_called_once_with("end", "new text")

    def test_long_preview_keeps_the_changing_phrase_visible(self):
        overlay = _overlay()
        overlay.text.count.return_value = 30
        updates = queue.SimpleQueue()
        updates.put(PreviewText("Long confirmed text", " changing phrase", 10.0))

        overlay.pump(updates)

        overlay.text.configure.assert_any_call(height=8)
        overlay.text.see.assert_called_once_with("end")

    def test_initial_layout_waits_for_a_real_width_before_counting_lines(self):
        overlay = _overlay()
        overlay.text.winfo_width.return_value = 1

        overlay._resize()

        overlay.text.count.assert_not_called()
        overlay.text.configure.assert_called_once_with(height=1)


if __name__ == "__main__":
    unittest.main()
