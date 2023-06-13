# Class for extract features
import datetime
import pandas as pd
from typing import List, Union
from State import SCIPState
import numpy as np


class BaseFeaturizer:
    def __init__(self, problem_instance_file: str) -> None:
        self.model = SCIPState(problem_instance_file)
        self.model.get_model().hideOutput()
        self.model.get_model().readProblem(problem_instance_file)


class ContextExtractor(BaseFeaturizer):
    def __init__(self, problem_instance_file: str) -> None:
        super().__init__(problem_instance_file)

    @staticmethod
    def log(log_message, logfile=None):
        log_message = '[{}] {}'.format(datetime.datetime.now(), log_message)
        print(log_message)
        if logfile is not None:
            with open(logfile, mode='a') as f:
                print(log_message, file=f)

    @staticmethod
    def convert_dict_to_series(dict_data, prefix):
        series_data = {}
        for key, value in dict_data.items():
            if isinstance(value, dict) and 'values' in value:
                for i, sublist in enumerate(value['values']):
                    for j, subvalue in enumerate(sublist):
                        series_data[f'{prefix}_{key}_{i}_{j}'] = subvalue
            else:
                series_data[f'{prefix}_{key}'] = value

        return pd.Series(series_data)

    def extract_context(self, state: SCIPState) -> Union[List[Union[int, float]], np.ndarray, pd.Series, pd.DataFrame]:
        """
        Extracts context features from MIP instances

        Returns
        -------
        context_features : Union[List[Union[int, float]], np.ndarray, pd.Series, pd.DataFrame]
            The extracted context features
        """
        # Extract state features
        state.objective()
        state_data = self.extract_state({})

        # Extract variable, constraints, and edge features
        obj_value = pd.Series([state_data['objective_value']], name='objective_value')
        variable_features = self.extract_variable_features()
        constraint_features = self.extract_constraint_features()
        # edge_features = self.convert_dict_to_series(state['edge_features'], 'edge')

        # Create a DataFrame with the desired format for MABSelector
        # context_df = pd.concat([obj_value, variable_features, constraint_features, edge_features], axis=1)
        context_df = pd.concat([obj_value, variable_features, constraint_features], axis=1)
        context_df.fillna(0, inplace=True)
        context_df.reset_index(drop=True, inplace=True)

        return context_df

    def extract_state(self, buffer=None):
        """
        Extracts the state from the model

        Parameters
        ----------
        buffer : dict, optional
            Buffer dictionary to store states, by default None

        Returns
        -------
        state : dict
            Dictionary containing the state
        """
        if buffer is None:
            buffer = {}

        # Extract objective value
        if self.model.get_model().getStatus() in ['optimal', 'bestsollimit']:
            buffer['objective_value'] = self.model.get_model().getObjVal()
        else:
            buffer['objective_value'] = None

        # Extract variable features
        buffer['variable_features'] = self.extract_variable_features()

        # Extract constraint features
        buffer['constraint_features'] = self.extract_constraint_features()

        # Extract edge features
        # buffer['edge_features'] = self.extract_edge_features()

        return buffer  # return the buffer as the state

    def extract_variable_features(self):
        # Get variables and their properties from the model
        varbls = self.model.get_model().getVars()
        var_types = [v.vtype() for v in varbls]
        coefs = [v.getObj() for v in varbls]
        lbs = [v.getLbGlobal() for v in varbls]
        ubs = [v.getUbGlobal() for v in varbls]

        type_mapping = {"BINARY": 0, "INTEGER": 1, "IMPLINT": 2, "CONTINUOUS": 3}  # Add more mappings if needed
        var_types_numeric = [type_mapping.get(t, 0) for t in var_types]

        variable_features = pd.DataFrame({
            'type': var_types_numeric,  # Use the converted numeric representation
            'coef': coefs,
            'lb': lbs,
            'ub': ubs
        })

        variable_features = variable_features.astype({'type': int, 'coef': float, 'lb': float, 'ub': float})

        return variable_features

    def extract_constraint_features(self):
        conss = self.model.get_model().getConss()
        constraint_data = []
        max_length = 0

        for c in conss:
            if self.is_set_ppc_constraint(c):
                lhs, rhs, coefs = self.extract_setppc_constraint_value(c)
            else:
                lhs = self.model.get_model().getLhs(c)
                rhs = self.model.get_model().getRhs(c)
                coefs = self.model.get_model().getValsLinear(c)

            if isinstance(lhs, float):
                lhs = [lhs]
            if isinstance(rhs, float):
                rhs = [rhs]
            if isinstance(coefs, float):
                coefs = [coefs]

            max_length = max(max_length, len(lhs), len(rhs), len(coefs))
            constraint_data.append((lhs, rhs, coefs))

        lhss = np.zeros((len(constraint_data), max_length))
        rhss = np.zeros((len(constraint_data), max_length))
        cons_coefs = np.zeros((len(constraint_data), max_length))

        for i, (lhs, rhs, coef) in enumerate(constraint_data):
            lhss[i, :len(lhs)] = lhs
            rhss[i, :len(rhs)] = rhs
            coefs_dict = coef
            sorted_vars = sorted(coefs_dict.keys())
            cons_coefs[i, :len(sorted_vars)] = [coefs_dict.get(var_name, 0.0) for var_name in sorted_vars]

        constraint_features = pd.DataFrame({
            'lhs': lhss.flatten(),
            'rhs': rhss.flatten(),
            'cons_coefs': cons_coefs.flatten()
        })

        constraint_features = constraint_features.astype({'lhs': float, 'rhs': float, 'cons_coefs': float})

        return constraint_features

    def is_set_ppc_constraint(self, constraint):
        rhs = self.model.get_model().getRhs(constraint)
        lhs = self.model.get_model().getLhs(constraint)

        if self.model.get_model().isEQ(rhs, lhs):
            return rhs == 1 and lhs == 1  # Set partitioning constraint
        elif not self.model.get_model().isInfinity(rhs):
            return rhs == 1  # Set packing constraint
        elif not self.model.get_model().isInfinity(-lhs):
            return rhs == 1  # Set covering constraint

        return False

    # def check_constraint_direction(self, constraint, rhs, direction):
    #     lhs = self.model.get_model().getLhs(constraint)
    #
    #     if self.model.get_model().isEQ(rhs, lhs):
    #         return direction == '=='
    #     elif not self.model.get_model().isInfinity(rhs):
    #         return direction == '<='
    #     elif not self.model.get_model().isInfinity(-lhs):
    #         return direction == '>='
    #
    #     return False

    def extract_setppc_constraint_value(self, constraint):
        vars = self.model.get_model().getVars(constraint)
        rhs = 1.0
        lhs = 0.0
        coefs = {}
        coef_values = self.model.get_model().getValsLinear(constraint)
        for var in vars:
            var_type = var.vtype()
            if var_type == "BINARY":
                coefs[var.name] = 1.0
            else:
                if var.name in coef_values:
                    coefs[var.name] = coef_values[var.name]
                else:
                    coefs[var.name] = 0.0
            lhs += var.getObj() * coefs[var.name]

        return lhs, rhs, coefs