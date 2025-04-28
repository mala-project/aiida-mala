"""
Calculations provided by aiida_mala.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""

from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob, CalcJobProcessSpec

# TestNetworkParameters = DataFactory("mala.test_network")


class TestNetworkCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapping testing trained models.
    """

    _DEFAULT_INPUT_FILE = "aiida.in"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        spec.input("metadata.options.input_filename", valid_type=str, default=cls._DEFAULT_INPUT_FILE)

        spec.input("model", valid_type=orm.SinglefileData, help="The trained model file.")
        spec.input("input_data", valid_type=orm.FolderData, help="Specify the folder with input data.")
        spec.input("output_data", valid_type=orm.FolderData, help="Specify the folder with output data.")
        spec.input("te_snapshots", valid_type=orm.List, help="List of testing snapshots.")

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        # spec.inputs["metadata"]["options"]["parser_name"].default = "mala.train_network"

        # spec.output("model", valid_type=orm.SinglefileData, help="The trained model file.")

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
            self.inputs.te_snapshots,
        ]
        local_copy_list = []

        input_file_content = self._generate_input_file(*arguments)
        with folder.open(self.metadata.options.input_filename, "w") as handle:
            handle.write(input_file_content)

        codeinfo = datastructures.CodeInfo()
        # codeinfo.cmdline_params = self.inputs.parameters.cmdline_params(
        #     input_file=self.inputs.input_file.filename
        # )
        codeinfo.cmdline_params = [self.metadata.options.input_filename]

        codeinfo.code_uuid = self.inputs.code.uuid

        local_copy_list = []
        local_copy_list.append((self.inputs.model.uuid, self.inputs.model.filename, self.inputs.model.filename))
        for snapshot in self.inputs.te_snapshots.get_list():
            local_copy_list.append((self.inputs.input_data.uuid, f"{snapshot}.in.npy", f"{snapshot}.in.npy"))
            local_copy_list.append((self.inputs.output_data.uuid, f"{snapshot}.out.npy", f"{snapshot}.out.npy"))
            local_copy_list.append((self.inputs.output_data.uuid, f"{snapshot}.info.json", f"{snapshot}.info.json"))

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = local_copy_list
        # calcinfo.retrieve_list = ["Be_model.zip"]

        return calcinfo

    @classmethod
    def _generate_input_file(cls, te_snapshots):  # pylint: disable=invalid-name
        """Create the input file"""

        input_file = ""
        input_file += "import os\n"
        input_file += "import mala\n"
        input_file += "from mala import printout\n"

        model_name = "Be_model"
        model_path = "./"
        input_file += (
            "parameters, network, data_handler, tester ="
            f" mala.Tester.load_run(run_name='{model_name:s}', path='{model_path:s}')\n"
        )
        input_file += "tester.observables_to_test = ['band_energy', 'density']\n"
        input_file += "tester.output_format = 'list'\n"
        input_file += "parameters.data.use_lazy_loading = True\n"

        for snapshot in te_snapshots.get_list():
            input_file += (
                f"data_handler.add_snapshot('{snapshot:s}.in.npy', '.',"
                f" '{snapshot:s}.out.npy', '.', 'te',"
                f" calculation_output_file=os.path.join('.', '{snapshot:s}.info.json'),)\n"
            )

        input_file += "data_handler.prepare_data(reparametrize_scaler=False)\n"

        input_file += "results = tester.test_all_snapshots()\n"

        return input_file
