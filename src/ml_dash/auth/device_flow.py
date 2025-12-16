"""Device authorization flow (RFC 8628) client for ml-dash."""

import time
from dataclasses import dataclass
from typing import Optional

import httpx

from .constants import VUER_AUTH_URL, CLIENT_ID, DEFAULT_SCOPE
from .device_secret import hash_device_secret
from .exceptions import (
    DeviceCodeExpiredError,
    AuthorizationDeniedError,
    TokenExchangeError,
)


@dataclass
class DeviceFlowResponse:
    """Response from device flow initiation."""

    user_code: str
    device_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


class DeviceFlowClient:
    """Client for OAuth 2.0 Device Authorization Flow (RFC 8628)."""

    def __init__(self, device_secret: str, ml_dash_server_url: str):
        """Initialize device flow client.

        Args:
            device_secret: Persistent device secret for this client
            ml_dash_server_url: ML-Dash server URL for token exchange
        """
        self.device_secret = device_secret
        self.ml_dash_server_url = ml_dash_server_url.rstrip("/")

    def start_device_flow(self, scope: str = DEFAULT_SCOPE) -> DeviceFlowResponse:
        """Initiate device authorization flow with vuer-auth.

        Args:
            scope: OAuth scopes to request

        Returns:
            DeviceFlowResponse with user code and verification URI

        Raises:
            httpx.HTTPError: If request fails
        """
        response = httpx.post(
            f"{VUER_AUTH_URL}/api/device-flow/start",
            json={
                "client_id": CLIENT_ID,
                "scope": scope,
                "device_secret_hash": hash_device_secret(self.device_secret),
            },
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        return DeviceFlowResponse(
            user_code=data["user_code"],
            device_code=data.get("device_code", ""),
            verification_uri=data["verification_uri"],
            verification_uri_complete=data.get(
                "verification_uri_complete",
                f"{data['verification_uri']}?code={data['user_code'].replace('-', '')}"
            ),
            expires_in=data.get("expires_in", 600),
            interval=data.get("interval", 5),
        )

    def poll_for_token(
        self,
        max_attempts: int = 120,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Poll vuer-auth for authorization completion.

        Args:
            max_attempts: Maximum polling attempts (default: 120 = 10 minutes at 5s intervals)
            progress_callback: Optional callback(elapsed_seconds) for progress updates

        Returns:
            Vuer-auth access token (JWT)

        Raises:
            DeviceCodeExpiredError: If device code expires
            AuthorizationDeniedError: If user denies authorization
            TimeoutError: If polling times out
        """
        device_secret_hash = hash_device_secret(self.device_secret)

        for attempt in range(max_attempts):
            elapsed = attempt * 5  # 5 second intervals

            if progress_callback:
                progress_callback(elapsed)

            try:
                response = httpx.post(
                    f"{VUER_AUTH_URL}/api/device-flow/poll",
                    json={
                        "client_id": CLIENT_ID,
                        "device_secret_hash": device_secret_hash,
                    },
                    timeout=10.0,
                )

                if response.status_code == 200:
                    # Authorization successful
                    data = response.json()
                    return data["access_token"]

                # Check error responses
                error_data = response.json()
                error = error_data.get("error")

                if error == "authorization_pending":
                    # Still waiting for user authorization
                    time.sleep(5)
                    continue
                elif error == "expired_token":
                    raise DeviceCodeExpiredError(
                        "Device code expired. Please run 'ml-dash login' again."
                    )
                elif error == "access_denied":
                    raise AuthorizationDeniedError(
                        "User denied authorization request."
                    )
                elif error == "slow_down":
                    # Server requests slower polling
                    time.sleep(10)
                    continue
                else:
                    # Unknown error
                    raise TokenExchangeError(f"Device flow error: {error}")

            except httpx.HTTPError as e:
                # Network error, retry
                time.sleep(5)
                continue

        raise TimeoutError(
            "Authorization timed out after 10 minutes. Please run 'ml-dash login' again."
        )

    def exchange_token(self, vuer_auth_token: str) -> str:
        """Exchange vuer-auth token for ml-dash permanent token.

        This calls the ml-dash server's token exchange endpoint.
        The server will:
        1. Decode the vuer-auth JWT
        2. Validate signature and expiry
        3. Extract username from claims
        4. Generate a permanent ml-dash token for that username
        5. Return the ml-dash token

        Args:
            vuer_auth_token: Temporary vuer-auth JWT access token

        Returns:
            Permanent ml-dash token string

        Raises:
            TokenExchangeError: If exchange fails
            httpx.HTTPError: If request fails
        """
        try:
            response = httpx.post(
                f"{self.ml_dash_server_url}/api/auth/exchange",
                headers={"Authorization": f"Bearer {vuer_auth_token}"},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            ml_dash_token = data.get("ml_dash_token")
            if not ml_dash_token:
                raise TokenExchangeError(
                    "Server response missing ml_dash_token field"
                )

            return ml_dash_token

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise TokenExchangeError(
                    "Vuer-auth token invalid or expired. Please try logging in again."
                )
            elif e.response.status_code == 404:
                raise TokenExchangeError(
                    "Token exchange endpoint not found. "
                    "Please ensure ml-dash server is up to date."
                )
            else:
                raise TokenExchangeError(
                    f"Token exchange failed: {e.response.status_code} {e.response.text}"
                )
        except httpx.HTTPError as e:
            raise TokenExchangeError(f"Network error during token exchange: {e}")

    def authenticate(
        self, progress_callback: Optional[callable] = None
    ) -> str:
        """Complete full device authorization flow.

        This is a convenience method that:
        1. Starts device flow with vuer-auth
        2. Polls for authorization
        3. Exchanges vuer-auth token for ml-dash token

        Args:
            progress_callback: Optional callback(elapsed_seconds) for progress updates

        Returns:
            Permanent ml-dash token string

        Raises:
            Various authentication exceptions
        """
        # Step 1: Start device flow
        flow = self.start_device_flow()

        # Step 2: Poll for authorization (caller should display user_code)
        vuer_auth_token = self.poll_for_token(progress_callback=progress_callback)

        # Step 3: Exchange for ml-dash token
        ml_dash_token = self.exchange_token(vuer_auth_token)

        return ml_dash_token
