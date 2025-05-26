"""Launch a calculation using the 'mala' plugin"""

import os.path
import sys
from pathlib import Path

from aiida import engine, orm
from aiida.common.exceptions import NotExistent
from aiida.plugins import DataFactory
from mala.datahandling.data_repo import data_path  # type: ignore

INPUT_DIR = Path(__file__).resolve().parent

# Create or load code
computer = orm.load_computer("localhost")
try:
    code = orm.load_code("mala.test_network@localhost")
except NotExistent:
    # Setting up code via python API (or use "verdi code setup")
    code = orm.InstalledCode(
        label="mala.test_network",
        computer=computer,
        filepath_executable=sys.executable,
        default_calc_job_plugin="mala.test_network",
    )

# Set up inputs
builder = code.get_builder()


FolderData = DataFactory("core.folder")
input_data = FolderData(tree=data_path)
builder.input_data = input_data
output_data = FolderData(tree=data_path)
builder.output_data = output_data

model = os.path.join(data_path, "Be_model.zip")
builder.model = orm.SinglefileData(model)

te_snapshots = orm.List(["Be_snapshot2", "Be_snapshot3"])
builder.te_snapshots = te_snapshots

observables = orm.List(["band_energy", "density"])
builder.observables = observables


builder.metadata.description = "Test job submission with the aiida_test_network plugin"  # type: ignore

# Run the calculation & parse results
engine.run(builder)
# result = engine.run(builder)
# computed_diff = result['mala.preprocess_data"'].get_content()
# print(f'Computed diff between files:\n{computed_diff}')
