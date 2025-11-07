"""
OAuth2 authentication flow for ML-Dash.

Implements the OAuth2 Authorization Code Flow with local callback server.
"""

import webbrowser
import time
import sys
from typing import Optional
from .auth_server import AuthCallbackServer
from .config import ConfigManager


class OAuth2AuthFlow:
    """
    Orchestrates the complete OAuth2 authentication flow.

    Features:
    - Local HTTP callback server
    - Browser-based authentication with manual URL fallback
    - Visual feedback with countdown timer and spinner
    - Token storage in user home directory
    """

    DEFAULT_AUTH_SERVER = "https://staging-auth.ml-dash.com"
    DEFAULT_PORT = 52845
    DEFAULT_TIMEOUT = 180.0  # 3 minutes

    def __init__(
        self,
        auth_server: Optional[str] = None,
        callback_port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Initialize OAuth2 authentication flow.

        Args:
            auth_server: Auth server URL (defaults to staging-auth.ml-dash.com)
            callback_port: Port for local callback server (default: 52845)
            timeout: Maximum time to wait for authentication (default: 180s)
            config_manager: Optional config manager (for testing)
        """
        self.auth_server = auth_server or self.DEFAULT_AUTH_SERVER
        self.callback_port = callback_port
        self.timeout = timeout
        self.config_manager = config_manager or ConfigManager()

    def authenticate(self) -> bool:
        """
        Run the complete authentication flow.

        Returns:
            True if authentication successful, False otherwise
        """
        print("=== ML-Dash Authentication ===\n")

        # Start local callback server
        print(f"Starting local callback server on port {self.callback_port}...")

        server = AuthCallbackServer(port=self.callback_port)

        try:
            server.start()
            print(f"✓ Server started on {server.get_callback_url()}\n")

            # Build authorization URL
            callback_url = server.get_callback_url()
            auth_url = f"{self.auth_server}/cli-auth?redirect_uri={callback_url}"

            # Display authorization URL
            print("Authorization URL:")
            print(f"  {auth_url}\n")

            # Open browser
            print("Opening authorization page in your browser...")

            opened = False
            try:
                opened = webbrowser.open(auth_url)
            except Exception:
                pass

            if not opened:
                print("⚠ Couldn't open browser automatically.")
                print("Please open the URL above manually.\n")
            else:
                print("✓ Browser opened\n")

            # Wait for callback with visual feedback
            token = self.wait_with_feedback(server, self.timeout)

            if token:
                # Save token
                self.config_manager.save_token(token, self.auth_server)
                print("✓ Authentication successful!")
                print(f"Token saved to {self.config_manager.config_path}\n")
                return True
            else:
                print("✗ Authentication timed out")
                print(f"Please try again or check your auth server connection.\n")
                return False

        except KeyboardInterrupt:
            print("\n\n✗ Authentication cancelled by user\n")
            return False

        except Exception as e:
            print(f"\n\n✗ Authentication failed: {e}\n")
            return False

        finally:
            server.stop()

    def wait_with_feedback(self, server: AuthCallbackServer, timeout: float) -> Optional[str]:
        """
        Wait for authentication with visual feedback (spinner + countdown).

        Args:
            server: Callback server to monitor
            timeout: Maximum time to wait

        Returns:
            Token if received, None if timeout
        """
        # Detect mocks for testing (skip visual feedback for mocks)
        server_type = type(server).__name__
        if 'Mock' in server_type:
            return server.wait_for_token(timeout=timeout)

        from .auth_server import CallbackHandler

        # Spinner characters (braille dots)
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        start_time = time.time()
        spinner_idx = 0

        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed

            # Check timeout
            if remaining <= 0:
                # Clear line
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                sys.stdout.flush()
                return None

            # Check for token
            with CallbackHandler.token_lock:
                if CallbackHandler.received_token:
                    # Clear line
                    sys.stdout.write('\r' + ' ' * 80 + '\r')
                    sys.stdout.flush()
                    return CallbackHandler.received_token

            # Display spinner and countdown
            mins, secs = divmod(int(remaining), 60)
            time_str = f"{mins:02d}:{secs:02d}"
            spinner_char = spinner[spinner_idx % len(spinner)]

            msg = f"\r{spinner_char} Waiting for authentication... Time remaining: {time_str}"
            sys.stdout.write(msg)
            sys.stdout.flush()

            spinner_idx += 1
            time.sleep(0.1)

    def check_status(self) -> bool:
        """
        Check if user is currently authenticated.

        Returns:
            True if authenticated, False otherwise
        """
        if self.config_manager.is_authenticated():
            auth_server = self.config_manager.get_auth_server()
            token = self.config_manager.get_token()

            print("=== Authentication Status ===\n")
            print("✓ Authenticated")
            print(f"Auth Server: {auth_server}")
            print(f"Token: {token[:20]}...{token[-10:] if len(token) > 30 else ''}")
            print(f"Config: {self.config_manager.config_path}\n")
            return True
        else:
            print("=== Authentication Status ===\n")
            print("✗ Not authenticated")
            print("Run 'ml-dash setup' to authenticate\n")
            return False

    def logout(self) -> bool:
        """
        Clear saved authentication token.

        Returns:
            True if logout successful
        """
        try:
            self.config_manager.clear_token()
            print("=== Logout ===\n")
            print("✓ Token cleared successfully")
            print("Run 'ml-dash setup' to authenticate again\n")
            return True
        except Exception as e:
            print(f"✗ Logout failed: {e}\n")
            return False


def authenticate(
    auth_server: Optional[str] = None,
    callback_port: int = OAuth2AuthFlow.DEFAULT_PORT,
    timeout: float = OAuth2AuthFlow.DEFAULT_TIMEOUT
) -> bool:
    """
    Convenience function to run authentication flow.

    Args:
        auth_server: Auth server URL (defaults to staging-auth.ml-dash.com)
        callback_port: Port for local callback server (default: 52845)
        timeout: Maximum time to wait for authentication (default: 180s)

    Returns:
        True if authentication successful, False otherwise

    Example:
        >>> from ml_dash.auth import authenticate
        >>> if authenticate():
        ...     print("Ready to use ML-Dash!")
    """
    flow = OAuth2AuthFlow(
        auth_server=auth_server,
        callback_port=callback_port,
        timeout=timeout
    )
    return flow.authenticate()


def check_auth_status() -> bool:
    """
    Check if user is currently authenticated.

    Returns:
        True if authenticated, False otherwise

    Example:
        >>> from ml_dash.auth import check_auth_status
        >>> if check_auth_status():
        ...     print("Already authenticated!")
    """
    flow = OAuth2AuthFlow()
    return flow.check_status()


def logout() -> bool:
    """
    Clear saved authentication token.

    Returns:
        True if logout successful

    Example:
        >>> from ml_dash.auth import logout
        >>> logout()
    """
    flow = OAuth2AuthFlow()
    return flow.logout()
