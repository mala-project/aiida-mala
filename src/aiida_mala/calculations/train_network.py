"""
Calculations provided by aiida_mala.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""

from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.plugins import DataFactory

TrainNetworkParameters = DataFactory("mala.train_network")


class TrainNetworkCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapping training of the data.
    """

    _DEFAULT_INPUT_FILE = "aiida.in"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        spec.input("metadata.options.input_filename", valid_type=str, default=cls._DEFAULT_INPUT_FILE)

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            help="The input parameters that are to be used to construct the input file.",
        )

        spec.input("input_data", valid_type=orm.FolderData, help="Specify the folder with input data.")
        spec.input("output_data", valid_type=orm.FolderData, help="Specify the folder with output data.")
        spec.input("tr_snapshots", valid_type=orm.List, help="List of training snapshots.")
        spec.input("va_snapshots", valid_type=orm.List, help="List of validation snapshots.")

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }

        spec.output("model", valid_type=orm.SinglefileData, help="The trained model file.")

        spec.exit_code(
            300,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Calculation did not produce all expected output files.",
        )

    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        arguments = [
            self.inputs.parameters,
            self.inputs.tr_snapshots,
            self.inputs.va_snapshots,
        ]
        local_copy_list = []

        input_file_content = self._generate_input(*arguments)
        with folder.open(self.metadata.options.input_filename, "w") as handle:
            handle.write(input_file_content)

        codeinfo = datastructures.CodeInfo()
        # codeinfo.cmdline_params = self.inputs.parameters.cmdline_params(
        #     input_file=self.inputs.input_file.filename
        # )
        codeinfo.cmdline_params = [self.metadata.options.input_filename]

        codeinfo.code_uuid = self.inputs.code.uuid

        local_copy_list = []
        for snapshot in self.inputs.tr_snapshots.get_list():
            local_copy_list.append((self.inputs.input_data.uuid, f"{snapshot}.in.npy", f"{snapshot}.in.npy"))
            local_copy_list.append((self.inputs.output_data.uuid, f"{snapshot}.out.npy", f"{snapshot}.out.npy"))
        for snapshot in self.inputs.va_snapshots.get_list():
            local_copy_list.append((self.inputs.input_data.uuid, f"{snapshot}.in.npy", f"{snapshot}.in.npy"))
            local_copy_list.append((self.inputs.output_data.uuid, f"{snapshot}.out.npy", f"{snapshot}.out.npy"))

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = local_copy_list
        calcinfo.retrieve_list = ["Be_model.zip"]

        return calcinfo

    @classmethod
    def _generate_input(cls, parameters: orm.Dict, tr_snapshots, va_snapshots):  # pylint: disable=invalid-name
        """Create the input file"""

        par_dict = parameters.get_dict()

        input_file = ""
        input_file += "import os\n"
        input_file += "import mala\n"

        input_file += "parameters = mala.Parameters()\n"
        for group in par_dict:
            for key, value in par_dict[group].items():
                if isinstance(value, str):
                    input_file += f'parameters.{group:s}.{key:s} = "{value:s}"\n'
                else:
                    input_file += f"parameters.{group:s}.{key:s} = {value}\n"

        input_file += "data_handler = mala.DataHandler(parameters)\n"

        for snapshot in tr_snapshots.get_list():
            input_file += f"data_handler.add_snapshot('{snapshot:s}.in.npy', '.', '{snapshot:s}.out.npy', '.', 'tr')\n"
        for snapshot in va_snapshots.get_list():
            input_file += f"data_handler.add_snapshot('{snapshot:s}.in.npy', '.', '{snapshot:s}.out.npy', '.', 'va')\n"

        input_file += "data_handler.prepare_data()\n"

        input_file += "parameters.network.layer_sizes = [\n"
        input_file += "    data_handler.input_dimension,\n"
        input_file += "    100,\n"
        input_file += "    data_handler.output_dimension,\n"
        input_file += "]\n"
        input_file += "test_network = mala.Network(parameters)\n"

        input_file += "test_trainer = mala.Trainer(parameters, test_network, data_handler)\n"
        input_file += "test_trainer.train_network()\n"
        # input_file += f"additional_calculation_data = os.path.join(data_path, \"Be_snapshot0.out\")\n"
        input_file += "test_trainer.save_run('Be_model')\n"
        # input_file += f"    \"Be_model\", additional_calculation_data=additional_calculation_data\n"

        return input_file
