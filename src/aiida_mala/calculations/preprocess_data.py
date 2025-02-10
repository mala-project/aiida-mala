"""
Calculations provided by aiida_mala.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""

from aiida.common import datastructures
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import SinglefileData
from aiida.plugins import DataFactory

PreprocessDataParameters = DataFactory("mala")


class PreprocessDataCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapping the diff executable.

    Simple AiiDA plugin wrapper for 'diffing' two files.
    """

    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        # spec.inputs["metadata"]["options"]["parser_name"].default = "mala"

        # new ports
        spec.input("metadata.options.output_filename", valid_type=str, default="snapshot.in")
        # spec.input(
        #     "parameters",
        #     valid_type=PreprocessDataParameters,
        #     help="Command line parameters for diff",
        # )
        # spec.input("ldos_file", valid_type=SinglefileData, help="LDOS cube file.")
        spec.input("input_file", valid_type=SinglefileData, help="MALA script file")
        # spec.output(
        #    "mala.preprocess_data",
        #    valid_type=SinglefileData,
        #    help="Preprocessed snapshot file.",
        # )

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
        codeinfo = datastructures.CodeInfo()
        # codeinfo.cmdline_params = self.inputs.parameters.cmdline_params(
        #    file1_name=self.inputs.input_file.filename
        # )
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.metadata.options.output_filename

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (
                self.inputs.input_file.uuid,
                self.inputs.input_file.filename,
                self.inputs.input_file.filename,
            ),
        ]
        calcinfo.retrieve_list = [self.metadata.options.output_filename]

        return calcinfo
