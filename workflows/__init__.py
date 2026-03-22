"""
DeepDriveSim is a Python package that provides a framework
for orchestrating AI-steered ensemble simulations.
It includes components for managing simulations, defining workflows,
and integrating with various simulation tools and machine learning models.
The package is designed to facilitate the development and execution of complex
simulation workflows, enabling researchers to efficiently explore and analyze
large parameter spaces in scientific computing domains.

"""


def _optional_import(path, name):
    try:
        module = __import__(path, fromlist=[name])
        return getattr(module, name)
    except Exception:
        return None


DDMdWorkflow = _optional_import("workflows.ddmd_workflow.ddmd_workflow", "DDMdWorkflow")
DummyWorkflow = _optional_import(
    "workflows.dummy_workflow.dummy_workflow", "DummyWorkflow"
)
MiniAppsWorkflow = _optional_import(
    "workflows.miniapps_workflow.miniapps_workflow", "MiniAppsWorkflow"
)
MiniAppsWorkflowAsyncflow = _optional_import(
    "workflows.miniapps_workflow.miniapps_workflow_asyncflow", "MiniAppsWorkflowAsyncflow"
)
AdaptiveWorkflow = _optional_import(
    "workflows.adaptive_ensemble.adaptive_workflow", "AdaptiveWorkflow"
)
__version__ = "0.1.0"

__all__ = [
    "DDMdWorkflow",
    "DummyWorkflow",
    "MiniAppsWorkflow",
    "AdaptiveWorkflow",
]
