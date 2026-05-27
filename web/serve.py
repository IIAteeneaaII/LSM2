"""
Servidor estático mínimo para probar la web localmente.

  python serve.py            -> http://localhost:8000
  python serve.py 8080       -> http://localhost:8080

La cámara solo se concede en orígenes seguros — localhost cuenta como seguro,
así que no necesitas HTTPS para probar.
"""
import http.server
import socketserver
import sys
import os

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DIR  = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js":   "application/javascript",
        ".mjs":  "application/javascript",
        ".onnx": "application/octet-stream",
        ".wasm": "application/wasm",
        ".task": "application/octet-stream",
    }

    def end_headers(self):
        # Necesario para que onnxruntime-web pueda usar SharedArrayBuffer si
        # hace falta (no estrictamente obligatorio, pero ayuda).
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        # Nunca cachear el modelo ni el label_map — si los reentrenas y
        # cambias clases, el browser caché de label_map viejo + ONNX nuevo
        # hace que el código crashee con labels undefined.
        if getattr(self, "path", "").startswith("/models/"):
            self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


os.chdir(DIR)
# ThreadingTCPServer + daemon_threads hace que Ctrl+C funcione en Windows
# (SimpleHTTPServer normal queda bloqueado en accept() y no procesa SIGINT).
class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads      = True

with Server(("", PORT), Handler) as httpd:
    print(f"Sirviendo {DIR} en http://localhost:{PORT}")
    print("Ctrl+C para detener.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDetenido.")
