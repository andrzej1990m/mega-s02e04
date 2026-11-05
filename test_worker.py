import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from worker import MCP, call_gemini, decode_rpc, has_receipt, main, unpack_result


class WorkerTests(unittest.TestCase):
    def test_gemini_retries_temporary_error_without_restarting_job(self):
        request = object()
        error = HTTPError("https://example.test", 503, "Unavailable", {}, io.BytesIO(b"{}"))
        self.addCleanup(error.close)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"candidates":[]}'
        log = []
        with (
            patch("worker.urllib.request.urlopen", side_effect=[error, response]) as urlopen,
            patch("worker.time.sleep"),
        ):
            self.assertEqual(call_gemini(request, log), {"candidates": []})
        self.assertEqual(urlopen.call_count, 2)
        self.assertTrue(all(call.args[0] is request for call in urlopen.call_args_list))
        self.assertEqual(log[0]["http_status"], 503)

    def test_gemini_stops_after_one_retry(self):
        error = HTTPError("https://example.test", 503, "Unavailable", {}, io.BytesIO(b"{}"))
        self.addCleanup(error.close)
        with (
            patch("worker.urllib.request.urlopen", side_effect=error) as urlopen,
            patch("worker.time.sleep"),
            self.assertRaises(HTTPError),
        ):
            call_gemini(object(), [])
        self.assertEqual(urlopen.call_count, 2)

    def test_mcp_identifies_application_in_http_request(self):
        with patch("worker.urllib.request.urlopen") as urlopen:
            response = urlopen.return_value.__enter__.return_value
            response.headers = {}
            response.read.return_value = b'{"jsonrpc":"2.0","id":1,"result":{}}'
            client = MCP("https://hq.mega.dev/mcp/job/test", [])
            client.request("initialize", {})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("User-agent"), "mega-incident-worker/1.0")

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
