import json
import unittest
from unittest.mock import patch

from worker import decode_rpc, has_receipt, main, unpack_result


class WorkerTests(unittest.TestCase):
    def test_decodes_json_and_sse(self):
        message = {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}
        data = json.dumps(message).encode()
        self.assertEqual(decode_rpc(data), message)
        self.assertEqual(
            decode_rpc(b": ping\r\n\r\nevent: message\r\ndata: " + data + b"\r\n\r\n"),
            message,
        )

    def test_finds_receipt_in_text_content(self):
        result = {"content": [{"type": "text", "text": '{"receipt":"example"}'}]}
        self.assertTrue(has_receipt(unpack_result(result)))

    def test_does_not_accept_text_mentioning_receipts(self):
        result = {"content": [{"type": "text", "text": "No receipt yet."}]}
        self.assertFalse(has_receipt(unpack_result(result)))

    def test_empty_model_lists_models_without_starting_job(self):
        with (
            patch.dict("os.environ", {"GEMINI_MODEL": ""}),
            patch("worker.list_models", return_value=["example-flash"]),
            patch("worker.run_agent") as run_agent,
            patch("worker.encrypt_result") as encrypt,
            patch("builtins.print"),
        ):
            main()
        run_agent.assert_not_called()
        self.assertEqual(encrypt.call_args.args[0]["status"], "models-listed")
        self.assertEqual(encrypt.call_args.args[0]["models"], ["example-flash"])


if __name__ == "__main__":
    unittest.main()
