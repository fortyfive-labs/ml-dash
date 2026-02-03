# ML-Dash Error Handling Audit

## Problem: Silent Error Handling

Currently, ml-dash catches and silently ignores many errors, making debugging difficult and hiding potential bugs.

## Categories of Issues

### 🔴 CRITICAL: Bare except: pass (Complete Silence)

**Location** | **Issue** | **Risk Level**
------------ | --------- | ------------
`upload.py:310` | `except: pass` | **CRITICAL** - Catches ALL exceptions including KeyboardInterrupt
`list.py:140` | `except:` returns timestamp | **HIGH** - Parsing errors hidden
`auto_start.py:72` | Cleanup errors ignored | **MEDIUM** - Resource leaks possible
`buffer.py:738` | Temp file cleanup ignored | **MEDIUM** - Disk space leaks
`remote_auto_start.py:53` | Cleanup errors ignored | **MEDIUM** - Resource leaks
`storage.py:82` | Lock cleanup ignored | **LOW** - Acceptable for cleanup
`auth/token_storage.py:257` | Key delete errors ignored | **LOW** - Acceptable for delete

### 🟠 HIGH: Return None/Empty on Error (Hidden Failures)

**Location** | **Issue** | **Impact**
------------ | --------- | ----------
`client.py:230` | `get_project_id()` returns None | Caller can't distinguish "not found" vs "error"
`client.py:268` | `get_current_user()` returns None | Silent authentication failures
`profile.py:62` | API call fails → return None | User sees no error message
`auth/token_storage.py:292` | Returns `{}` on exception | Empty config hides errors
`files.py:1274` | `exists()` returns False on error | File access errors hidden

### 🟡 MEDIUM: Warnings Instead of Errors (Non-Fatal but Problematic)

**Location** | **Issue** | **Problem**
------------ | --------- | -----------
`buffer.py:460-463` | Log flush failure → warning | Data loss not visible
`buffer.py:536-540` | Metric flush failure → warning | Silent data loss
`buffer.py:599-602` | Track flush failure → warning | Silent data loss
`experiment.py:403` | Status update failure → print | No proper logging
`experiment.py:557-559` | Log failure → warning | Data loss hidden
`experiment.py:1075-1077` | Param set failure → warning | Silent failure
`experiment.py:1101-1105` | Metric log failure → warning | Silent data loss

### 🟢 ACCEPTABLE: Cleanup/Fallback Errors

**Location** | **Reason** | **Acceptable?**
------------ | ---------- | ---------------
`storage.py:82` | Lock file cleanup | ✅ Yes - cleanup can fail safely
`buffer.py:738` | Temp file cleanup | ⚠️ Maybe - should log
`login.py:148` | Browser auto-open | ✅ Yes - user can manually open
`download.py:686` | Experiment not found in loop | ✅ Yes - expected in discovery

## Recommended Approach

### 1. Add Error Handling Modes

```python
class ErrorHandlingMode(Enum):
    """Controls how ml-dash handles non-critical errors."""
    STRICT = "strict"      # Raise all errors
    DEFAULT = "default"    # Warn on non-critical, raise on critical
    PERMISSIVE = "permissive"  # Warn on most errors, continue training
```

### 2. Classify Errors by Severity

```python
class MLDashError(Exception):
    """Base exception for ml-dash."""
    pass

class CriticalError(MLDashError):
    """Critical error that should always fail."""
    # Examples: Invalid credentials, corrupted data, API server unreachable
    pass

class DataLossError(MLDashError):
    """Error that could result in data loss."""
    # Examples: Failed to write logs, failed to save metrics
    pass

class RecoverableError(MLDashError):
    """Recoverable error that can be retried or worked around."""
    # Examples: Temporary network issue, buffer full
    pass
```

### 3. Smart Error Handling

```python
def handle_error(error: Exception, context: str, mode: ErrorHandlingMode):
    """
    Centralized error handling.

    Args:
        error: The exception
        context: What was being done (e.g., "flushing logs")
        mode: Error handling mode
    """
    if isinstance(error, CriticalError):
        # Always raise critical errors
        raise error

    if isinstance(error, DataLossError):
        if mode == ErrorHandlingMode.STRICT:
            raise error
        else:
            # Warn but don't crash training
            warnings.warn(
                f"Data loss risk in {context}: {error}\n"
                f"Set ML_DASH_ERROR_MODE=strict to fail on data loss errors",
                RuntimeWarning,
                stacklevel=2
            )

    if isinstance(error, RecoverableError):
        if mode == ErrorHandlingMode.STRICT:
            raise error
        elif mode == ErrorHandlingMode.DEFAULT:
            warnings.warn(f"Recoverable error in {context}: {error}", UserWarning)
        # PERMISSIVE: log but continue
```

### 4. Configuration

```python
# Environment variable
ML_DASH_ERROR_MODE = os.environ.get("ML_DASH_ERROR_MODE", "default")

# Or programmatic
from ml_dash import set_error_mode, ErrorHandlingMode
set_error_mode(ErrorHandlingMode.STRICT)  # Fail fast for debugging

# Or per-experiment
exp = Experiment(..., error_mode="strict")
```

## Priority Fixes

### P0 (Immediate) - Silent Data Loss

1. **buffer.py**: Flush failures should be more visible
   ```python
   # Current (BAD):
   except Exception as e:
       warnings.warn(f"Failed to flush: {e}")

   # Better:
   except Exception as e:
       if error_mode == "strict":
           raise DataLossError(f"Failed to flush logs: {e}")
       else:
           warnings.warn(f"⚠️ DATA LOSS RISK: Failed to flush logs: {e}", RuntimeWarning)
   ```

2. **upload.py:310**: Remove bare `except:`
   ```python
   # Current (VERY BAD):
   except:
       pass

   # Better:
   except (KeyError, ValueError) as e:
       logger.warning(f"Failed to parse prefix: {e}")
   ```

3. **metric.py**: Don't silently skip non-numeric values
   ```python
   # Current (BAD):
   except (TypeError, ValueError):
       continue  # Skip non-numeric values silently

   # Better:
   except (TypeError, ValueError) as e:
       if error_mode == "strict":
           raise ValueError(f"Non-numeric value for '{key}': {value}")
       else:
           logger.debug(f"Skipping non-numeric value for '{key}': {value}")
   ```

### P1 (High Priority) - Hidden Errors

1. **client.py**: Don't return None on errors
   ```python
   # Current (BAD):
   except Exception:
       return None

   # Better:
   except httpx.HTTPStatusError as e:
       if e.response.status_code == 404:
           return None  # Not found is expected
       raise  # Re-raise other errors
   except Exception as e:
       raise RemoteClientError(f"Failed to get project: {e}")
   ```

2. **files.py**: Don't hide file access errors
   ```python
   # Current (BAD):
   except Exception:
       return False

   # Better:
   except FileNotFoundError:
       return False  # Expected case
   except PermissionError as e:
       raise FileAccessError(f"Permission denied: {e}")
   except Exception as e:
       raise FileAccessError(f"Failed to check if file exists: {e}")
   ```

### P2 (Medium Priority) - Better Warnings

1. **experiment.py**: Use proper logging instead of print
   ```python
   # Current (BAD):
   except Exception as e:
       print(f"Warning: Failed to update experiment status: {e}")

   # Better:
   except Exception as e:
       logger.warning(f"Failed to update experiment status: {e}", exc_info=True)
   ```

2. **Add structured logging**:
   ```python
   import logging

   logger = logging.getLogger("ml_dash")
   logger.setLevel(logging.INFO)

   # In code:
   logger.error("Failed to flush metrics", exc_info=True)
   logger.warning("Retrying connection...")
   logger.info("Experiment started")
   logger.debug("Buffer contains 100 items")
   ```

## Implementation Plan

### Phase 1: Stop the Bleeding (Week 1)
- [ ] Fix bare `except:` clauses
- [ ] Add `DataLossError` warnings to buffer flush failures
- [ ] Remove return None patterns in critical paths
- [ ] Add proper logging infrastructure

### Phase 2: Error Classification (Week 2)
- [ ] Define exception hierarchy (CriticalError, DataLossError, RecoverableError)
- [ ] Add `ErrorHandlingMode` enum
- [ ] Implement centralized error handler

### Phase 3: Gradual Migration (Week 3-4)
- [ ] Update buffer.py to use new error handling
- [ ] Update experiment.py to use new error handling
- [ ] Update client.py to use new error handling
- [ ] Update files.py to use new error handling

### Phase 4: Testing & Documentation
- [ ] Add tests for each error mode
- [ ] Document error handling behavior
- [ ] Add migration guide for users

## Breaking Changes

⚠️ **Warning**: Some fixes will be breaking changes:

1. **Strict mode by default?** NO - keep permissive as default
2. **Return None → Raise?** YES for new code, NO for existing (use deprecation)
3. **Silent warnings → Loud warnings?** YES - add prominent warnings

## Backward Compatibility

To maintain compatibility:

1. Add opt-in strict mode (don't change default)
2. Use deprecation warnings before removing features
3. Document all behavior changes in CHANGELOG
4. Provide migration guide

## Example: Before & After

### Before (Silent Failure)
```python
try:
    client.flush_logs(logs)
except Exception as e:
    warnings.warn(f"Failed to flush: {e}")
    # Training continues, data is LOST
```

### After (Clear Error Handling)
```python
try:
    client.flush_logs(logs)
except httpx.NetworkError as e:
    # Transient error - retry
    warnings.warn(f"Network error, will retry: {e}")
    buffer.add_to_retry_queue(logs)
except httpx.HTTPStatusError as e:
    # Server error - may need user action
    if error_mode == "strict":
        raise DataLossError(f"Server rejected logs: {e}")
    else:
        warnings.warn(f"⚠️ DATA LOSS: Server rejected logs: {e}", RuntimeWarning)
except Exception as e:
    # Unexpected error - always raise
    raise DataLossError(f"Unexpected error flushing logs: {e}") from e
```

## Files to Update

### High Priority
- [ ] `src/ml_dash/buffer.py` - 8 locations
- [ ] `src/ml_dash/experiment.py` - 10 locations
- [ ] `src/ml_dash/client.py` - 3 locations
- [ ] `src/ml_dash/cli_commands/upload.py` - 1 CRITICAL bare except
- [ ] `src/ml_dash/metric.py` - 2 locations

### Medium Priority
- [ ] `src/ml_dash/files.py` - 10 locations
- [ ] `src/ml_dash/cli_commands/*.py` - Various locations
- [ ] `src/ml_dash/auth/token_storage.py` - 2 locations

### Low Priority (Cleanup)
- [ ] `src/ml_dash/auto_start.py` - Cleanup errors
- [ ] `src/ml_dash/storage.py` - Lock cleanup

## Questions for Discussion

1. **Default mode**: Should we default to `strict`, `default`, or `permissive`?
   - **Recommendation**: `default` - warn loudly but don't crash training

2. **Breaking changes**: When to introduce strict mode?
   - **Recommendation**: v0.7.0 with opt-in strict, v1.0.0 make it default

3. **Retry logic**: Should we add automatic retries for network errors?
   - **Recommendation**: Yes, with exponential backoff

4. **Buffering on error**: Should we queue failed operations for retry?
   - **Recommendation**: Yes, but with size limits
