# Change Log

## v0.1.0 (Planned)

Initial release of ML-Dash with basic functionality:

### Features
- Core logging API for metrics and parameters
- Hierarchical experiment organization with prefixes
- Async logging with local caching
- HTTP client for data-manager integration
- Artifact management (save/load)
- Support for model checkpoints and figures

### API
- `MLLogger` - Main logger class
- `log_params()` - Log hyperparameters
- `log_metrics()` - Log training metrics
- `save()` / `load()` - Artifact management
- `save_checkpoint()` - Model checkpoint saving
- `save_figure()` - Matplotlib figure saving

### Documentation
- Quick start guide
- API reference
- Installation instructions
