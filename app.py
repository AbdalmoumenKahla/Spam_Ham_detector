import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from predict_message import DEFAULT_MODEL_PATH, predict_text

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"


class SpamClassifierHandler(BaseHTTPRequestHandler):
	def do_GET(self) -> None:
		if self.path in {"/", "/index.html"}:
			self._serve_file("index.html", "text/html; charset=utf-8")
			return

		if self.path == "/styles.css":
			self._serve_file("styles.css", "text/css; charset=utf-8")
			return

		if self.path == "/app.js":
			self._serve_file("app.js", "application/javascript; charset=utf-8")
			return

		self.send_error(HTTPStatus.NOT_FOUND, "Not found")

	def do_POST(self) -> None:
		if self.path != "/api/predict":
			self.send_error(HTTPStatus.NOT_FOUND, "Not found")
			return

		content_length = int(self.headers.get("Content-Length", "0"))
		payload = self.rfile.read(content_length)

		try:
			data = json.loads(payload.decode("utf-8"))
		except json.JSONDecodeError:
			self._send_json({"error": "Request body must be valid JSON."}, HTTPStatus.BAD_REQUEST)
			return

		message = str(data.get("message", "")).strip()
		if not message:
			self._send_json({"error": "Message is required."}, HTTPStatus.BAD_REQUEST)
			return

		try:
			result = predict_text(message, DEFAULT_MODEL_PATH)
		except FileNotFoundError:
			self._send_json(
				{"error": "Model file not found. Train the model first with train_model.py."},
				HTTPStatus.INTERNAL_SERVER_ERROR,
			)
			return

		self._send_json(result)

	def log_message(self, format: str, *args: object) -> None:
		return

	def _serve_file(self, filename: str, content_type: str) -> None:
		file_path = UI_DIR / filename
		if not file_path.exists():
			self.send_error(HTTPStatus.NOT_FOUND, "Not found")
			return

		content = file_path.read_bytes()
		self.send_response(HTTPStatus.OK)
		self.send_header("Content-Type", content_type)
		self.send_header("Content-Length", str(len(content)))
		self.end_headers()
		self.wfile.write(content)

	def _send_json(self, data: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
		payload = json.dumps(data).encode("utf-8")
		self.send_response(status)
		self.send_header("Content-Type", "application/json; charset=utf-8")
		self.send_header("Content-Length", str(len(payload)))
		self.end_headers()
		self.wfile.write(payload)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run a local web app for spam message testing.")
	parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to.")
	parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to.")
	args = parser.parse_args()

	server = ThreadingHTTPServer((args.host, args.port), SpamClassifierHandler)
	print(f"Open http://{args.host}:{args.port} in your browser")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print("\nServer stopped.")
	finally:
		server.server_close()


if __name__ == "__main__":
	main()