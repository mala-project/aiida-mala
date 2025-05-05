"""
Parsers provided by aiida_mala.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""

import json

from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory

TestNetworkCalculation = CalculationFactory("mala.test_network")


class TestNetworkParser(Parser):
    """
    Parser class for parsing output of calculation.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a TestNetworkCalculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.nodes.process.process.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, TestNetworkCalculation):
            raise exceptions.ParsingError("Can only parse TestNetworkCalculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        # output_filename = self.node.get_option("output_filename")

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = ["observables.json"]
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(f"Found files '{files_retrieved}', expected to find '{files_expected}'")
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # add output file
        self.logger.info("Parsing 'observables.json'")
        with self.retrieved.open("observables.json", "r") as handle:
            output_node = Dict(json.load(handle))
        self.out("observables", output_node)

        return ExitCode(0)
