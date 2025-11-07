"""
Local HTTP callback server for OAuth2 authentication.

Implements a lightweight HTTP server that listens for OAuth callback redirects
and extracts the access token.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional
import threading
import time


class CallbackHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for OAuth callback.

    Handles GET requests on root path (/) and extracts access_token parameter.
    Thread-safe token storage with class-level attributes.
    """

    # Class-level attributes for thread-safe token storage
    received_token: Optional[str] = None
    token_lock = threading.Lock()

    def log_message(self, format, *args):
        """Suppress HTTP server logging."""
        pass

    def do_OPTIONS(self):
        """Handle OPTIONS request for CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self):
        """
        Handle GET request with OAuth callback.

        Extracts access_token from query parameters and stores it.
        Returns HTML success/error page to user's browser.
        """
        # Parse URL
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        # Extract token
        token = query_params.get('access_token', [None])[0]

        if token:
            # Store token in class attribute (thread-safe)
            with CallbackHandler.token_lock:
                CallbackHandler.received_token = token

            # Send success response
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.end_headers()

            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authentication Successful</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    }
                    .container {
                        background: white;
                        padding: 48px;
                        border-radius: 16px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 400px;
                    }
                    .success-icon {
                        font-size: 64px;
                        color: #10b981;
                        margin-bottom: 24px;
                    }
                    h1 {
                        color: #1f2937;
                        margin: 0 0 16px 0;
                        font-size: 28px;
                    }
                    p {
                        color: #6b7280;
                        margin: 0 0 24px 0;
                        font-size: 16px;
                        line-height: 1.5;
                    }
                    .code {
                        background: #f3f4f6;
                        padding: 12px 16px;
                        border-radius: 8px;
                        font-family: 'Monaco', 'Menlo', monospace;
                        font-size: 14px;
                        color: #374151;
                        margin-top: 16px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="success-icon">✓</div>
                    <h1>Authentication Successful!</h1>
                    <p>You have successfully authenticated with ML-Dash.</p>
                    <p>You can now close this window and return to your terminal.</p>
                    <div class="code">ml-dash is ready to use</div>
                </div>
            </body>
            </html>
            """

            self.wfile.write(html_content.encode('utf-8'))
        else:
            # Send error response
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.end_headers()

            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Authentication Failed</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
                    }
                    .container {
                        background: white;
                        padding: 48px;
                        border-radius: 16px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 400px;
                    }
                    .error-icon {
                        font-size: 64px;
                        color: #ef4444;
                        margin-bottom: 24px;
                    }
                    h1 {
                        color: #1f2937;
                        margin: 0 0 16px 0;
                        font-size: 28px;
                    }
                    p {
                        color: #6b7280;
                        margin: 0 0 24px 0;
                        font-size: 16px;
                        line-height: 1.5;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="error-icon">✗</div>
                    <h1>Authentication Failed</h1>
                    <p>No access token was received from the authentication server.</p>
                    <p>Please try again or contact support if the problem persists.</p>
                </div>
            </body>
            </html>
            """

            self.wfile.write(html_content.encode('utf-8'))


class AuthCallbackServer:
    """
    Local HTTP server for receiving OAuth callback.

    Starts a local HTTP server on port 52845 (with fallbacks to 52846, 52847)
    and waits for OAuth callback with access token.
    """

    def __init__(self, port: int = 52845):
        """
        Initialize callback server.

        Args:
            port: Port to listen on (default: 52845)
        """
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """
        Start the HTTP server in a background thread.

        Tries ports 52845, 52846, 52847 in order until one is available.

        Raises:
            RuntimeError: If all ports are in use
        """
        # Reset token state
        with CallbackHandler.token_lock:
            CallbackHandler.received_token = None

        # Try ports with fallback
        ports_to_try = [self.port, self.port + 1, self.port + 2]

        for port in ports_to_try:
            try:
                self.server = HTTPServer(('localhost', port), CallbackHandler)
                self.port = port  # Update to actual port used
                break
            except OSError as e:
                if port == ports_to_try[-1]:
                    raise RuntimeError(
                        f"Failed to start server on ports {ports_to_try}. "
                        f"Please ensure these ports are not in use."
                    )
                continue

        # Start server in background thread
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the HTTP server and clean up."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

        # Save thread reference before setting to None
        thread = self.thread
        self.thread = None

        # Wait for thread to finish if it exists
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

    def wait_for_token(self, timeout: float = 180.0) -> Optional[str]:
        """
        Wait for OAuth callback to receive token.

        Args:
            timeout: Maximum time to wait in seconds (default: 180)

        Returns:
            Token string if received, None if timeout
        """
        start_time = time.time()

        while True:
            # Check if token received
            with CallbackHandler.token_lock:
                if CallbackHandler.received_token:
                    return CallbackHandler.received_token

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return None

            # Small sleep to avoid busy waiting
            time.sleep(0.1)

    def get_callback_url(self) -> str:
        """
        Get the callback URL for this server.

        Returns:
            Full callback URL (e.g., http://localhost:52845)
        """
        return f"http://localhost:{self.port}"

    def __enter__(self):
        """Context manager entry - start server."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop server."""
        self.stop()
        return False
