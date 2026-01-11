"""
Demo: dxp and rdxp with manual start

Shows the new behavior where dxp and rdxp require manual start.
"""

from ml_dash import dxp

print("Demo: dxp with Manual Start")
print("=" * 60)

# Check initial state
print(f"\n1. After import:")
print(f"   dxp._is_open: {dxp._is_open}")
print(f"   ✓ dxp is NOT auto-started on import")

# Method 1: Using 'with' statement (recommended)
print(f"\n2. Using 'with dxp.run:' (recommended):")
with dxp.run:
    print(f"   Inside with block - dxp._is_open: {dxp._is_open}")
    dxp.params.set(method="with_statement", learning_rate=0.001)
    dxp.logs.info("Training with 'with' statement")
print(f"   After with block - dxp._is_open: {dxp._is_open}")
print(f"   ✓ Automatically completed on exit")

# Method 2: Manual start/complete
print(f"\n3. Manual start/complete:")
dxp.run.start()
print(f"   After start() - dxp._is_open: {dxp._is_open}")
dxp.params.set(method="manual", batch_size=32)
dxp.logs.info("Training with manual start/complete")
dxp.run.complete()
print(f"   After complete() - dxp._is_open: {dxp._is_open}")
print(f"   ✓ Manually controlled lifecycle")

print("\n" + "=" * 60)
print("Summary:")
print("  • dxp no longer auto-starts on import")
print("  • Must use 'with dxp.run:' or dxp.run.start()")
print("  • 'with' statement is recommended (auto-completes)")
print("  • Still has atexit cleanup if you forget to complete")
print("=" * 60)
