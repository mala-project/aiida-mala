"""Data types provided by plugin

Register data types via the "aiida.data" entry point in setup.json.
"""

# You can directly use or subclass aiida.orm.data.Data
# or any other data type listed under 'verdi data'
from aiida.orm import Dict
from voluptuous import Optional, Schema

# A subset of diff's command line options
cmdline_options = {
    Optional("ignore-case"): bool,
    Optional("ignore-file-name-case"): bool,
    Optional("ignore-tab-expansion"): bool,
    Optional("ignore-space-change"): bool,
    Optional("ignore-all-space"): bool,
}


class PreprocessDataParameters(Dict):  # pylint: disable=too-many-ancestors
    """
    Command line options for diff.

    This class represents a python dictionary used to
    pass command line options to the executable.
    """

    # "voluptuous" schema  to add automatic validation
    schema = Schema(cmdline_options)

    # pylint: disable=redefined-builtin
    def __init__(self, dict=None, **kwargs):
        """
        Constructor for the data class

        Usage: ``DiffParameters(dict{'ignore-case': True})``

        :param parameters_dict: dictionary with commandline parameters
        :param type parameters_dict: dict

        """
        dict = self.validate(dict)
        super().__init__(dict=dict, **kwargs)

    def validate(self, parameters_dict):
        """Validate command line options.

        Uses the voluptuous package for validation. Find out about allowed keys using::

            print(PreprocessDataParameters).schema.schema

        :param parameters_dict: dictionary with commandline parameters
        :param type parameters_dict: dict
        :returns: validated dictionary
        """
        return PreprocessDataParameters.schema(parameters_dict)

    def cmdline_params(self, input_file):
        """Synthesize command line parameters.

        e.g. [ '--ignore-case', 'inputfile']

        :param input_file: Name of first file
        :param type input_file: str

        """
        parameters = []

        pm_dict = self.get_dict()
        for option, enabled in pm_dict.items():
            if enabled:
                parameters += ["--" + option]

        parameters += [input_file]

        return [str(p) for p in parameters]

    def __str__(self):
        """String representation of node.

        Append values of dictionary to usual representation. E.g.::

            uuid: b416cbee-24e8-47a8-8c11-6d668770158b (pk: 590)
            {'ignore-case': True}

        """
        string = super().__str__()
        string += "\n" + str(self.get_dict())
        return string


class TrainNetworkParameters(Dict):  # pylint: disable=too-many-ancestors
    """
    Command line options for diff.

    This class represents a python dictionary used to
    pass command line options to the executable.
    """

    # "voluptuous" schema  to add automatic validation
    schema = Schema(cmdline_options)

    # pylint: disable=redefined-builtin
    def __init__(self, dict=None, **kwargs):
        """
        Constructor for the data class

        Usage: ``DiffParameters(dict{'ignore-case': True})``

        :param parameters_dict: dictionary with commandline parameters
        :param type parameters_dict: dict

        """
        dict = self.validate(dict)
        super().__init__(dict=dict, **kwargs)

    def validate(self, parameters_dict):
        """Validate command line options.

        Uses the voluptuous package for validation. Find out about allowed keys using::

            print(PreprocessDataParameters).schema.schema

        :param parameters_dict: dictionary with commandline parameters
        :param type parameters_dict: dict
        :returns: validated dictionary
        """
        return PreprocessDataParameters.schema(parameters_dict)

    def cmdline_params(self, input_file):
        """Synthesize command line parameters.

        e.g. [ '--ignore-case', 'inputfile']

        :param input_file: Name of first file
        :param type input_file: str

        """
        parameters = []

        pm_dict = self.get_dict()
        for option, enabled in pm_dict.items():
            if enabled:
                parameters += ["--" + option]

        parameters += [input_file]

        return [str(p) for p in parameters]

    def __str__(self):
        """String representation of node.

        Append values of dictionary to usual representation. E.g.::

            uuid: b416cbee-24e8-47a8-8c11-6d668770158b (pk: 590)
            {'ignore-case': True}

        """
        string = super().__str__()
        string += "\n" + str(self.get_dict())
        return string
