"""Data types provided by plugin

Register data types via the "aiida.data" entry point in setup.json.
"""

# You can directly use or subclass aiida.orm.data.Data
# or any other data type listed under 'verdi data'
from aiida.orm import Dict
from voluptuous import All, In, Optional, Required, Schema


class TrainNetworkParameters(Dict):  # pylint: disable=too-many-ancestors
    """
    Options for training the network.
    """

    data_schema = Schema(
        {
            Required("input_rescaling_type"): In(["feature-wise-standard", "minmax"]),
            Required("output_rescaling_type"): In(["feature-wise-standard", "minmax"]),
        }
    )
    network_schema = Schema(
        {
            Required("layer_activations"): All(list, [str]),
        }
    )
    running_schema = Schema(
        {
            Required("max_number_epochs"): int,
            Required("mini_batch_size"): int,
            Required("learning_rate"): float,
            Required("optimizer"): In(["Adam", "SGD"]),
        }
    )
    descriptors_schema = Schema(
        {
            Required("descriptor_type"): In(["Bispectrum", "SOAP"]),
            Optional("bispectrum_twojmax"): int,
            Optional("bispectrum_cutoff"): float,
        }
    )
    targets_schema = Schema(
        {
            Required("target_type"): In(["LDOS", "Energy"]),
            Optional("ldos_gridsize"): int,
            Optional("ldos_gridspacing_ev"): float,
            Optional("ldos_gridoffset_ev"): int,
        }
    )

    schema = Schema(
        {
            Required("data"): data_schema,
            Required("network"): network_schema,
            Required("running"): running_schema,
            Required("descriptors"): descriptors_schema,
            Required("targets"): targets_schema,
        }
    )

    # pylint: disable=redefined-builtin
    def __init__(self, dict=None, **kwargs):
        """
        Constructor for the data class

        Usage: ``TrainNetworkParameters(dict{'ignore-case': True})``

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
        return TrainNetworkParameters.schema(parameters_dict)

    def __str__(self):
        """String representation of node.

        Append values of dictionary to usual representation. E.g.::

            uuid: b416cbee-24e8-47a8-8c11-6d668770158b (pk: 590)
            {'ignore-case': True}

        """
        string = super().__str__()
        string += "\n" + str(self.get_dict())
        return string
