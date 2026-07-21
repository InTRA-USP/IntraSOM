import numpy as np
import inspect
import sys


small = .000000000001


class NeighborhoodFactory(object):

    """
    Class for creating the neighborhood function definition object.
    """

    @staticmethod
    def build(neighborhood_func):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and neighborhood_func == obj.name:
                    return obj()
        else:
            raise Exception(
                "Unsupported neighborhood function '%s'" % neighborhood_func)


class GaussianNeighborhood(object):

    """
    Creation of a Gaussian neighborhood function definition object.
    """

    name = 'gaussian'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        return np.exp(-1.0*distance_matrix/(2.0*radius**2)).reshape(dim, dim)

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)


class BubbleNeighborhood(object):

    name = 'bubble'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        def l(a, b):
            c = np.zeros(b.shape)
            c[a-b >= 0] = 1
            return c

        return l(radius,
                 np.sqrt(distance_matrix.flatten())).reshape(dim, dim) + small

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)
    


class NormalizerFactory(object):
    """
    Class for creating the data normalization object.
    """

    @staticmethod
    def build(type_name, **kwargs):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj(**kwargs)

        raise Exception(
            "Unknown normalization type '%s'" % type_name
        )


class Normalizer(object):

    """
    Class model for implementing new normalization objects.
    """

    name = "Object_name"

    def normalize(self, data):
        raise NotImplementedError()

    def normalize_by(self, raw_data, data):
        raise NotImplementedError()

    def denormalize_by(self, raw_data, data):
        raise NotImplementedError()

# Para criar outros normalizadores só copiar esse codigo e substituir
class VarianceNormalizer(Normalizer):

    """
    Normalization by variance. The data will be normalized by subtracting the mean and dividing by 
    the standard deviation of that variable.
    """

    name = 'var'

    def _mean_and_standard_dev(self, data):
        return np.nanmean(data, axis=0), np.nanstd(data, axis=0)

    def normalize(self, data):
        me, st = self._mean_and_standard_dev(data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return np.round((data-me)/st,10)

    def normalize_by(self, raw_data, data, with_labels=False, pred_size=None):
        if with_labels:
            me, st = self._mean_and_standard_dev(raw_data[:, : -pred_size])
            st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN

        else:
            me, st = self._mean_and_standard_dev(raw_data)
            st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN

        return np.round((data - me) / st,10)

    def denormalize_by(self, data_by, n_vect, with_labels=False, pred_size=None):
        if with_labels:
            me, st = self._mean_and_standard_dev(data_by[:, :(data_by.shape[1] - pred_size)])
        else:
            me, st = self._mean_and_standard_dev(data_by)
        return np.round(n_vect * st + me,10)
    
class NoneNormalizer(Normalizer):
    """
    Class for handling cases when normalization should not be applied.

    The method signatures are kept compatible with the other
    normalizers so that the same projection and denormalization
    routines can be used independently of the selected normalization.
    """

    name = "None"

    def normalize(self, data):
        return data

    def normalize_by(
        self,
        raw_data,
        data,
        with_labels=False,
        pred_size=None
    ):
        return data

    def denormalize_by(
        self,
        raw_data,
        data,
        with_labels=False,
        pred_size=None
    ):
        return data

class WeightedVarianceNormalizer(VarianceNormalizer):
    """
    Normalização por variância com pesos por variável.

    Primeiro aplica z-score:

        z = (x - média) / desvio

    Depois aplica:

        z_weighted = z * sqrt(peso)

    Isso equivale a usar uma distância Euclidiana ponderada,
    em que o valor informado em `weights` representa diretamente
    a contribuição relativa de cada variável na distância quadrática.

    Exemplos
    --------
    weights = [1, 1, 1]
        Todas as variáveis têm o mesmo peso.

    weights = [2, 1, 1]
        A primeira variável contribui 2x mais para a distância.

    weights = [0.5, 1, 1]
        A primeira variável contribui metade das demais.
    """

    name = "var_weighted"

    def __init__(self, weights=None):
        self.weights = weights

    def _get_weight_factors(
        self,
        n_features,
        allow_subset=False
    ):
        """
        Returns sqrt(weights) for the requested number of features.

        Parameters
        ----------
        n_features : int
            Number of variables that will be normalized.

        allow_subset : bool, default=False
            If False, the number of weights must exactly match
            n_features.

            If True, the first n_features weights may be used.
            This is required when projecting only the first variables
            of a model trained with additional label columns.
        """

        if self.weights is None:
            return np.ones(
                n_features,
                dtype=float
            )

        weights = np.asarray(
            self.weights,
            dtype=float
        ).reshape(-1)

        # Validate all supplied weights
        if np.any(~np.isfinite(weights)):
            raise ValueError(
                "Todos os pesos devem ser valores finitos."
            )

        if np.any(weights <= 0):
            raise ValueError(
                "Todos os pesos devem ser maiores que zero."
            )

        # --------------------------------------------------
        # EXACT MATCH
        # Used during normal training
        # --------------------------------------------------

        if len(weights) == n_features:

            selected_weights = weights

        # --------------------------------------------------
        # PREFIX SUBSET
        # Used when projecting fewer variables
        # --------------------------------------------------

        elif (
            allow_subset
            and n_features < len(weights)
        ):

            selected_weights = weights[
                :n_features
            ]

        else:

            raise ValueError(
                "Número incorreto de pesos. "
                f"Foram fornecidos {len(weights)} pesos "
                f"e são necessárias {n_features} variáveis."
            )

        return np.sqrt(
            selected_weights
        )

    def normalize(self, data):

        data = np.asarray(data, dtype=float)

        me, st = self._mean_and_standard_dev(data)

        st = np.asarray(st, dtype=float)
        st[st == 0] = 1

        normalized = (data - me) / st

        factors = self._get_weight_factors(
            data.shape[1]
        )

        weighted = normalized * factors

        return np.round(weighted, 10)

    def normalize_by(
        self,
        raw_data,
        data,
        with_labels=False,
        pred_size=None
    ):

        raw_data = np.asarray(
            raw_data,
            dtype=float
        )

        data = np.asarray(
            data,
            dtype=float
        )

        if with_labels:

            if pred_size is None:
                raise ValueError(
                    "`pred_size` deve ser informado "
                    "quando `with_labels=True`."
                )

            reference_data = raw_data[:, :-pred_size]

        else:

            reference_data = raw_data

        me, st = self._mean_and_standard_dev(
            reference_data
        )

        st = np.asarray(st, dtype=float)
        st[st == 0] = 1

        if data.shape[-1] != len(me):
            raise ValueError(
                "O número de variáveis de `data` não coincide "
                "com o número de variáveis usadas para normalização. "
                f"data={data.shape[-1]}, esperado={len(me)}."
            )

        normalized = (data - me) / st

        factors = self._get_weight_factors(
            len(me),
            allow_subset=True
        )

        weighted = normalized * factors

        return np.round(weighted, 10)

    def denormalize_by(
        self,
        data_by,
        n_vect,
        with_labels=False,
        pred_size=None
    ):

        data_by = np.asarray(
            data_by,
            dtype=float
        )

        n_vect = np.asarray(
            n_vect,
            dtype=float
        )

        if with_labels:

            if pred_size is None:
                raise ValueError(
                    "`pred_size` deve ser informado "
                    "quando `with_labels=True`."
                )

            reference_data = data_by[:, :-pred_size]

        else:

            reference_data = data_by

        me, st = self._mean_and_standard_dev(
            reference_data
        )

        st = np.asarray(st, dtype=float)
        st[st == 0] = 1

        factors = self._get_weight_factors(
            len(me),
            allow_subset=True
        )

        # Primeiro remove os pesos
        normalized = n_vect / factors

        # Depois desfaz o z-score
        original = normalized * st + me

        return np.round(original, 10)