.PHONY: build-docs docs preview help

help:
	@echo "Available targets:"
	@echo "  build-docs  - Build the Sphinx documentation"
	@echo "  docs        - Build and serve documentation with auto-reload"
	@echo "  preview     - Build and preview documentation in browser"

build-docs:
	@echo "Building documentation..."
	uv run sphinx-build -M html docs docs/_build

docs: build-docs
	@echo "Starting documentation server at http://127.0.0.1:8001"
	@echo "Press Ctrl+C to stop the server"
	cd docs/_build/html && uv run python -m http.server 8001

preview: build-docs
	@echo "Starting documentation server with auto-reload..."
	@echo "Documentation will be available at http://127.0.0.1:8001"
	uv run sphinx-autobuild docs docs/_build/html --port 8001 --open-browser
