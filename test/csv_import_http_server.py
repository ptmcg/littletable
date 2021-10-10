from http.server import HTTPServer, BaseHTTPRequestHandler
from http import HTTPStatus
import threading


# noinspection PyPep8Naming
class CSVTestRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.log_message("received %s %s", self.command, self.path)
        path = self.path
        if path == "/EXIT":
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            threading.Thread(target=lambda: self.server.shutdown()).start()

        if path.startswith("/abc.csv"):
            send_bytes = b"a,b,c\n1,2,3\n"
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            self.wfile.write(send_bytes)


def run(server_class=HTTPServer, handler_class=CSVTestRequestHandler):
    server_address = ('', 8888)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()


if __name__ == '__main__':
    run()
