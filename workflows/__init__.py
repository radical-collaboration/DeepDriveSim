"""
DeepDriveSim is a Python package that provides a framework 
for orchestrating AI-steered ensemble simulations. 
It includes components for managing simulations, defining workflows, 
and integrating with various simulation tools and machine learning models. 
The package is designed to facilitate the development and execution of complex 
simulation pipelines, enabling researchers to efficiently explore and analyze 
large parameter spaces in scientific computing domains.

"""

def _optional_import(path, name):
    try:
        module = __import__(path, fromlist=[name])
        return getattr(module, name)
    except ModuleNotFoundError:
        return None


DDMdWorkflow = _optional_import(
    "pipelines.ddmd_pipeline.ddmd_pipeline", "DDMdWorkflow"
)
DummyWorkflow = _optional_import(
    "pipelines.dummy_pipeline.dummy_pipeline", "DummyWorkflow"
)
MiniAppsWorkflow = _optional_import(
    "pipelines.miniapps_pipeline.miniapps_pipeline", "MiniAppsWorkflow"
)

__version__ = "0.1.0"

__all__ = [
    "DDMdWorkflow",
    "DummyWorkflow",
    "MiniAppsWorkflow",
]