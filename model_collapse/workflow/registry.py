"""Lazy CLI capabilities. Importing the registry loads neither Torch nor cloud clients."""
from importlib import import_module


_ROUTES = {
    "paired": {
        "worker": "workflow.paired:run_pair",
        "report": "analysis.paired:report_pair",
        "validate": "workflow.paired:validate_pair_local",
    },
    "mixed_campaign": {
        "worker": "workflow.mixed_campaign:run_campaign",
        "report": "analysis.campaign:report_campaign",
        "validate": "continued:validate_campaign_local",
    },
    "fresh_real": {
        "worker": "workflow.fresh_real:run",
        "report": "analysis.fresh_real:report",
        "validate": "workflow.fresh_validation:validate_local",
    },
    "completion_comparison": {
        "worker": "workflow.completion:run",
        "report": "analysis.completion:report",
        "validate": "workflow.completion_validation:validate_local",
    },
    "completion_arm": {
        "worker": "workflow.completion:run_arm",
        # Retain the historical generic single-arm report/validation behavior.
        "report": "analysis.reporting:report",
        "validate": "continued:validate_local",
    },
}


def known_workflow(name):
    return name is None or (isinstance(name, str) and name in _ROUTES)


def handler(c, operation):
    if operation not in ("worker", "report", "validate"):
        raise ValueError(f"Unknown workflow capability: {operation}")
    workflow = c.get("workflow")
    if workflow is None:
        routes = {"worker": "workflow.wikitext:run", "report": "analysis.reporting:report",
                  "validate": "continued:validate_local" if c.get("initialization") in ("continued", "reset_s0") else "validate:validate"}
    else:
        if workflow not in _ROUTES:
            raise ValueError(f"Unknown workflow: {workflow}")
        routes = _ROUTES[workflow]
    module, name = routes[operation].split(":")
    return getattr(import_module("model_collapse." + module), name)
