from typing import Protocol

import numpy as np
from pandas import DataFrame


class State(Protocol):
    """
    Protocol for a solution state. Solutions should define an ``objective()``
    member function for evaluation.
    """

    def objective(self) -> float:
        """
        Computes the state's associated objective value.
        """


class ContextualState(State, Protocol):
    """
    Protocol for a solution state that also provides context. Solutions should
    define ``objective()`` and ``get_context()`` methods.
    """

    def get_context(self) -> np.ndarray:
        """
        Computes a context vector for the current state.
        """


class StateSCIP:
    """
    Solution class for the MIP problem implemented using SCIP.
    """

    def __init__(self, model):
        self.model = model

    def objective(self) -> float:
        """
        Computes the objective value of the current solution.
        """
        if self.model.getStatus() in ['optimal', 'bestsollimit']:
            return self.model.getObjVal()
        else:
            return float('nan')

    def get_mip_context(self) -> np.ndarray:
        """
        Computes a context vector for the current solution.
        Here you can extract relevant features or information from the SCIP model to form the context.
        """
        # Extract variable features
        varbls = self.model.getVars()
        var_types = [v.vtype() for v in varbls]
        coefs = [v.getObj() for v in varbls]
        lbs = [v.getLbGlobal() for v in varbls]
        ubs = [v.getUbGlobal() for v in varbls]

        type_mapping = {"BINARY": 0, "INTEGER": 1, "IMPLINT": 2, "CONTINUOUS": 3}
        var_types_numeric = [type_mapping.get(t, 0) for t in var_types]

        variable_features = DataFrame({
            'type': var_types_numeric,
            'coef': coefs,
            'lb': lbs,
            'ub': ubs
        })

        variable_features = variable_features.astype({'type': int, 'coef': float, 'lb': float, 'ub': float})

        # Extract constraint features
        conss = self.model.getConss()
        constraint_data = []
        max_length = 0

        for c in conss:
            lhs = self.model.getLhs(c)
            rhs = self.model.getRhs(c)
            coefs = self.model.getValsLinear(c)

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
            cons_coefs[i, :len(coef)] = coef

        constraint_features = DataFrame({
            'lhs': lhss.flatten(),
            'rhs': rhss.flatten(),
            'cons_coefs': cons_coefs.flatten()
        })

        constraint_features = constraint_features.astype({'lhs': float, 'rhs': float, 'cons_coefs': float})

        # Concatenate variable and constraint features
        context_features = DataFrame(variable_features)
        context_features = context_features.join(constraint_features, how='outer')
        context_features.fillna(0, inplace=True)

        # Convert to numpy array
        context = context_features.values

        return context
