"""M12: Studio Frontend integration tests.

Tests verify:
- Frontend build output exists and is correctly structured
- Server serves the built frontend (not placeholder)
- SPA routes return index.html (catch-all)
- API endpoints still work with frontend mounted
- Static assets are served correctly
"""

import json
from pathlib import Path

import pytest

# ---- Project structure tests ----


class TestFrontendProjectStructure:
    """Verify the React project scaffold exists."""

    project_dir = Path(__file__).parent.parent / "studio-frontend"

    def test_project_directory_exists(self):
        assert self.project_dir.exists(), "studio-frontend/ directory should exist"

    def test_package_json_exists(self):
        pkg = self.project_dir / "package.json"
        assert pkg.exists()
        data = json.loads(pkg.read_text())
        assert "react" in data.get("dependencies", {})
        assert "react-router-dom" in data.get("dependencies", {})
        assert "@tanstack/react-query" in data.get("dependencies", {})
        assert "recharts" in data.get("dependencies", {})
        assert "lucide-react" in data.get("dependencies", {})

    def test_vite_config_exists(self):
        assert (self.project_dir / "vite.config.ts").exists()

    def test_src_directory_structure(self):
        src = self.project_dir / "src"
        assert (src / "App.tsx").exists()
        assert (src / "main.tsx").exists()
        assert (src / "index.css").exists()
        assert (src / "api" / "client.ts").exists()
        assert (src / "api" / "types.ts").exists()

    def test_hooks_exist(self):
        hooks = self.project_dir / "src" / "hooks"
        assert (hooks / "useRuns.ts").exists()
        assert (hooks / "useModels.ts").exists()
        assert (hooks / "useDatasets.ts").exists()
        assert (hooks / "useTraining.ts").exists()
        assert (hooks / "useWebSocket.ts").exists()

    def test_pages_exist(self):
        pages = self.project_dir / "src" / "pages"
        assert (pages / "Dashboard.tsx").exists()
        assert (pages / "Experiments.tsx").exists()
        assert (pages / "RunDetail.tsx").exists()
        assert (pages / "Models.tsx").exists()
        assert (pages / "Datasets.tsx").exists()
        assert (pages / "Playground.tsx").exists()
        assert (pages / "Settings.tsx").exists()

    def test_layout_components_exist(self):
        layout = self.project_dir / "src" / "components" / "layout"
        assert (layout / "Sidebar.tsx").exists()
        assert (layout / "PageLayout.tsx").exists()

    def test_shared_components_exist(self):
        shared = self.project_dir / "src" / "components" / "shared"
        assert (shared / "StatCard.tsx").exists()
        assert (shared / "StatusBadge.tsx").exists()
        assert (shared / "MetricCard.tsx").exists()

    def test_chart_components_exist(self):
        charts = self.project_dir / "src" / "components" / "charts"
        assert (charts / "LossChart.tsx").exists()

    def test_playground_components_exist(self):
        playground = self.project_dir / "src" / "components" / "playground"
        assert (playground / "ChatMessage.tsx").exists()
        assert (playground / "ChatInput.tsx").exists()
        assert (playground / "GenerationConfig.tsx").exists()


# ---- Built output tests ----


class TestFrontendBuildOutput:
    """Verify the production build was copied to the backend package."""

    frontend_dir = Path(__file__).parent.parent / "cortexlab" / "studio" / "frontend"

    def test_frontend_directory_exists(self):
        assert self.frontend_dir.exists(), "cortexlab/studio/frontend/ should exist"

    def test_index_html_exists(self):
        index = self.frontend_dir / "index.html"
        assert index.exists()
        content = index.read_text()
        assert "<script" in content, "index.html should include script tags"
        assert "/assets/" in content, "index.html should reference assets"

    def test_assets_directory_exists(self):
        assets = self.frontend_dir / "assets"
        assert assets.exists()

    def test_js_bundle_exists(self):
        assets = self.frontend_dir / "assets"
        js_files = list(assets.glob("*.js"))
        assert len(js_files) > 0, "Should have at least one JS bundle"

    def test_css_bundle_exists(self):
        assets = self.frontend_dir / "assets"
        css_files = list(assets.glob("*.css"))
        assert len(css_files) > 0, "Should have at least one CSS bundle"


# ---- Server integration tests ----


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    try:
        from cortexlab.studio.server import create_app
    except ImportError:
        pytest.skip("FastAPI not installed")
    return create_app()


@pytest.fixture
def client(app):
    """Create a test client."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("starlette not installed")
    return TestClient(app)


class TestServerServesFrontend:
    """Verify the server serves the built frontend."""

    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_root_returns_spa_html(self, client):
        resp = client.get("/")
        content = resp.text
        assert "<script" in content, "Should serve SPA with script tags"
        # Should NOT be the placeholder
        assert "Frontend not built" not in content

    def test_spa_route_returns_index_html(self, client):
        """SPA routes should return index.html for client-side routing."""
        resp = client.get("/experiments")
        assert resp.status_code == 200
        assert "<script" in resp.text

    def test_nested_spa_route_returns_index_html(self, client):
        resp = client.get("/experiments/some-run-id")
        assert resp.status_code == 200
        assert "<script" in resp.text

    def test_assets_served(self, client):
        """Static assets should be accessible."""
        frontend_dir = Path(__file__).parent.parent / "cortexlab" / "studio" / "frontend"
        assets = list((frontend_dir / "assets").glob("*.js"))
        if assets:
            asset_name = assets[0].name
            resp = client.get(f"/assets/{asset_name}")
            assert resp.status_code == 200


class TestApiStillWorks:
    """Verify API endpoints still work with frontend mounted."""

    def test_runs_endpoint(self, client):
        resp = client.get("/api/v1/runs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_models_supported_endpoint(self, client):
        resp = client.get("/api/v1/models/supported")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_datasets_endpoint(self, client):
        resp = client.get("/api/v1/datasets")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_training_active_endpoint(self, client):
        resp = client.get("/api/v1/training/active")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_docs_endpoint(self, client):
        """Swagger UI should still be accessible."""
        resp = client.get("/docs")
        assert resp.status_code == 200
