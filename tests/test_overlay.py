import queue
import unittest
from sys import modules
from unittest.mock import Mock, patch

from voicecommander.local_asr import PreviewText
from voicecommander.preview import PreviewOverlay


def _overlay():
    overlay = PreviewOverlay.__new__(PreviewOverlay)
    overlay.root = Mock()
    overlay.text = Mock()
    return overlay


class PreviewOverlayTests(unittest.TestCase):
    def test_preview_overlay_consumes_worker_updates(self) -> None:
        tkinter = Mock()
        tkinter.Tk.return_value.winfo_screenwidth.return_value = 1920
        tkinter.Text.return_value.winfo_reqwidth.return_value = 300
        tkinter.Text.return_value.winfo_reqheight.return_value = 40
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
        text.configure.assert_any_call(state="disabled")
        self.assertEqual(tkinter.Text.call_args.kwargs["height"], 4)
        tkinter.Tk.return_value.deiconify.assert_called_once()
        tkinter.Tk.return_value.update.assert_not_called()

    def test_pump_shows_latest_text_hides_and_reopens(self):
        overlay = _overlay()
        updates = queue.SimpleQueue()
        updates.put("old")
        updates.put("latest")
        overlay.pump(updates)
        overlay.text.insert.assert_called_once_with("end", "latest")
        overlay.root.deiconify.assert_called_once_with()

        updates.put("stale")
        updates.put(None)
        overlay.pump(updates)
        overlay.root.withdraw.assert_called_once_with()
        overlay.root.deiconify.assert_called_once_with()
        overlay.text.insert.assert_called_once_with("end", "latest")

        updates.put("new text")
        overlay.pump(updates)
        self.assertEqual(overlay.root.deiconify.call_count, 2)
        self.assertEqual(
            [call.args for call in overlay.text.insert.call_args_list],
            [("end", "latest"), ("end", "new text")],
        )

    def test_pump_empty_queue_leaves_overlay_unchanged(self):
        overlay = _overlay()

        overlay.pump(queue.SimpleQueue())

        overlay.root.withdraw.assert_not_called()
        overlay.root.deiconify.assert_not_called()
        overlay.text.configure.assert_not_called()

    def test_preview_shows_a_bounded_single_paragraph_and_preserves_tentative_style(self):
        cases = (
            (PreviewText("  Hello\n\n", "\t world  \n", 1.0), "Hello", " world"),
            (
                PreviewText("obsolete " * 100 + "confirmed", "\n newest words", 10.0),
                None,
                " newest words",
            ),
            (PreviewText("obsolete", " word" * 200 + " newest words", 10.0), "", None),
            ("old " * 200 + "latest\n\nrefinement\t words", None, None),
        )
        for update, stable, tentative in cases:
            with self.subTest(update=type(update).__name__, stable=stable):
                overlay = _overlay()
                updates = queue.SimpleQueue()
                updates.put(update)

                overlay.pump(updates)

                inserted = [call.args[1] for call in overlay.text.insert.call_args_list]
                displayed = "".join(inserted)
                original = update.text if isinstance(update, PreviewText) else update
                normalized = " ".join(original.split())
                self.assertLessEqual(len(displayed), 500)
                self.assertTrue(normalized.endswith(displayed))
                self.assertTrue(displayed.endswith(normalized[-20:]))
                self.assertEqual(displayed, " ".join(displayed.split()))
                if isinstance(update, PreviewText):
                    if stable is not None:
                        self.assertEqual(inserted[0], stable)
                    if tentative is not None:
                        self.assertEqual(inserted[1], tentative)
                    self.assertEqual(overlay.text.insert.call_args.args[2], "tentative")
                overlay.text.see.assert_called_once_with("end")
                overlay.text.count.assert_not_called()


if __name__ == "__main__":
    unittest.main()
