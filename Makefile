.PHONY: help docs sync-skill check-skill

help:
	@echo "Available targets:"
	@echo "  docs         - Open the documentation site"
	@echo "  sync-skill   - Refresh skills/dash-docs/ from docs.dash.ml"
	@echo "  check-skill  - Fail if the vendored skill is stale"
	@echo ""
	@echo "Documentation lives at https://docs.dash.ml and is built from"
	@echo "dreamlake-ai/dash-workspace (docs/). It is no longer built here —"
	@echo "the Sphinx tree and the Docusaurus site have both been retired."

docs:
	@echo "https://docs.dash.ml"
	@command -v open >/dev/null && open https://docs.dash.ml || true

sync-skill:
	./scripts/sync-docs-skill.sh

check-skill:
	./scripts/sync-docs-skill.sh --check
