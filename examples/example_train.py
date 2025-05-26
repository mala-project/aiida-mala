"""Launch a calculation using the 'mala' plugin"""

import sys
from pathlib import Path

from aiida import engine, orm
from aiida.common.exceptions import NotExistent
from aiida.plugins import DataFactory
from mala.datahandling.data_repo import data_path  # type: ignore

TrainNetworkParameters = DataFactory("mala.train_network")

INPUT_DIR = Path(__file__).resolve().parent

# Create or load code
computer = orm.load_computer("localhost")
try:
    code = orm.load_code("mala.train_network@localhost")
except NotExistent:
    # Setting up code via python API (or use "verdi code setup")
    code = orm.InstalledCode(
        label="mala.train_network",
        computer=computer,
        filepath_executable=sys.executable,
        default_calc_job_plugin="mala.train_network",
    )

# Set up inputs
builder = code.get_builder()

parameters = {
    "data": {
        "input_rescaling_type": "feature-wise-standard",
        "output_rescaling_type": "minmax",
    },
    "network": {
        "layer_activations": ["ReLU"],
    },
    "running": {
        "max_number_epochs": 100,
        "mini_batch_size": 40,
        "learning_rate": 0.00001,
        "optimizer": "Adam",
    },
    "descriptors": {
        "descriptor_type": "Bispectrum",
        "bispectrum_twojmax": 10,
        "bispectrum_cutoff": 4.67637,
    },
    "targets": {
        "target_type": "LDOS",
        "ldos_gridsize": 11,
        "ldos_gridspacing_ev": 2.5,
        "ldos_gridoffset_ev": -5,
    },
}
builder.parameters = TrainNetworkParameters(parameters)

FolderData = DataFactory("core.folder")
input_data = FolderData(tree=data_path)
builder.input_data = input_data
output_data = FolderData(tree=data_path)
builder.output_data = output_data

tr_snapshots = orm.List(["Be_snapshot0"])
builder.tr_snapshots = tr_snapshots
va_snapshots = orm.List(["Be_snapshot1"])
builder.va_snapshots = va_snapshots

builder.metadata.description = "Test job submission with the aiida_train_network plugin"  # type:ignore

# Run the calculation & parse results
engine.run(builder)
