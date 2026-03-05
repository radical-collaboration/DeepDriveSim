from typing import Optional

from workflows.ddmd_workflow.config import AggregationTaskConfig


class BasicAggegation(AggregationTaskConfig):
    rmsd: bool = True
    fnc: bool = False
    contact_map: bool = False
    point_cloud: bool = True
    verbose: bool = True
    last_n_h5_files: Optional[int]


if __name__ == "__main__":
    BasicAggegation().dump_yaml("basic_aggregation_template.yaml")
