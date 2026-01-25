"""Debug script to check dxp configuration."""
from ml_dash.auto_start import dxp

print("=" * 60)
print("DXP Configuration Debug")
print("=" * 60)

# Check operation mode
print(f"Operation Mode: {dxp.mode}")
print(f"dash_url from Experiment: {getattr(dxp, '_dash_url', 'Not set')}")
print(f"dash_root from Experiment: {getattr(dxp, '_dash_root', 'Not set')}")

# Check RUN configuration
print(f"\nRUN.api_url: {dxp.run.api_url}")
print(f"RUN.user: {dxp.run.user}")
print(f"RUN.project: {dxp.run.project}")
print(f"RUN.prefix: {dxp.run.prefix}")

# Check client and storage
print(f"\nRUN._client: {dxp.run._client}")
print(f"RUN._storage: {dxp.run._storage}")

print(f"\nIs open: {dxp._is_open}")

# Try to use it
print("\n" + "=" * 60)
print("Testing with context manager...")
print("=" * 60)

try:
    with dxp.run:
        print(f"After start - Operation Mode: {dxp.mode}")
        print(f"After start - _client: {dxp.run._client}")
        print(f"After start - _storage: {dxp.run._storage}")
        dxp.log("Test log message")
        print("✓ Log succeeded")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
