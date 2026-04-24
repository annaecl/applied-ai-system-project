import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests that make real Gemini API calls (requires GOOGLE_API_KEY).",
    )


@pytest.fixture
def live_mode(request):
    return request.config.getoption("--live")
