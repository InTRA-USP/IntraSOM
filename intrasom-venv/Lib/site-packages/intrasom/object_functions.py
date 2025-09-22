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
    def build(type_name):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj()
        else:
            raise Exception("Unknown normalization type '%s'" % type_name)


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
    """

    name = "None"

    def normalize(self, data):
        return data

    def normalize_by(self, raw_data, data):
        return data

    def denormalize_by(self, raw_data, data):
        return data

