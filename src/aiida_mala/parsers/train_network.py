"""
Parsers provided by aiida_mala.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import SinglefileData
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

TrainNetworkCalculation = CalculationFactory("mala.train_network")


class PreprocessDataParser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a TrainNetworkCalculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.nodes.process.process.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, TrainNetworkCalculation):
            raise exceptions.ParsingError("Can only parse TrainNetworkCalculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        # output_filename = self.node.get_option("output_filename")

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = ["Be_model.zip"]
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(f"Found files '{files_retrieved}', expected to find '{files_expected}'")
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # add output file
        self.logger.info(f"Parsing '{'Be_model.zip'}'")
        with self.retrieved.open("Be_model.zip", "rb") as handle:
            output_node = SinglefileData(file=handle)
        self.out("mala.train_network", output_node)

        return ExitCode(0)
