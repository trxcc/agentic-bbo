"""Container-only loopback-to-Unix relay, using Python's standard library."""
import select
import socket
import socketserver
import subprocess
import sys
import threading


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect("/run/bbo-model.sock")
            peers = {self.request: upstream, upstream: self.request}
            while True:
                readable, _, _ = select.select(list(peers), [], [], 30)
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    peers[source].sendall(data)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    with Server(("127.0.0.1", 38080), Relay) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            code = subprocess.call(sys.argv[1:])
        finally:
            server.shutdown()
        raise SystemExit(code)
