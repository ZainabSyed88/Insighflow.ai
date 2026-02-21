"""Simple HTTP server to serve the landing page."""

import http.server
import socketserver
import os
import webbrowser
import threading

PORT = 8500
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landing")


class LandingHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files from the landing directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass


def open_browser():
    """Open the landing page in the default browser after a short delay."""
    import time
    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}")


def main():
    print(f"\n{'='*52}")
    print(f"  🔬 AI Data Insights Analyst — Landing Page")
    print(f"{'='*52}")
    print(f"  Landing page:  http://localhost:{PORT}")
    print(f"  Streamlit app: http://localhost:8501")
    print(f"{'='*52}")
    print(f"  Press Ctrl+C to stop\n")

    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    with socketserver.TCPServer(("", PORT), LandingHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
            httpd.server_close()


if __name__ == "__main__":
    main()
