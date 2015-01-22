import numpy as np
from numpy.random import randint, rand
import scipy.io
from scipy.optimize import minimize


IMAGE_FILE = "IMAGES.mat"
IMAGE_MAT_TAG = 'IMAGES'
PATCH_SIZE = (8, 8)
PATCH_SIZE_1D = PATCH_SIZE[0] * PATCH_SIZE[1]
N_PATCHES_TRAIN = 10000


def sample_training_images(whitening=True, recenter_std=3.0):
    """Load all the images and randomly sample patches from it.

    Parameters
    ----------
    whitening : If true, the data is whitened to have zero mean and
    unit standard deviation.
    recenter_std : Use with whitening. Recenter the data at 0.5. data at
    +recenter_std shifts to 1, and data at -recenter_std shifts to 0.

    Returns
    -------
    A training data matrix containing the image patches sampled.
    The dimension is (N_PATCHES_TRAIN, PATCH_SIZE_1D).
    """
    # Load the image. Here it is a 512*512*10 numpy array.
    all_images = scipy.io.loadmat(IMAGE_FILE)[IMAGE_MAT_TAG]
    image_dimension = all_images.shape
    training_data = []
    for i in xrange(N_PATCHES_TRAIN):
        image_id = randint(image_dimension[2])
        patch_x_start = randint(image_dimension[0] - PATCH_SIZE[0])
        patch_x_end = patch_x_start + PATCH_SIZE[0]
        patch_y_start = randint(image_dimension[1] - PATCH_SIZE[1])
        patch_y_end = patch_y_start + PATCH_SIZE[1]

        # Slice out the patch
        patch = all_images[patch_x_start:patch_x_end,
                           patch_y_start:patch_y_end,
                           image_id]
        # Flatten the patch and append
        training_data.append(np.ravel(patch))

    if whitening:
        training_data -= np.mean(training_data)
        training_data /= np.std(training_data)
        training_data += recenter_std
        training_data /= recenter_std * 2

    return np.asarray(training_data)


# The following codes assume a 3-layer sparse autoencoder.
class NetworkParams(object):
    """Parameters for the neural network. The parameters are internally stored
    as a 1d array. The values for parameters at each layer can be set and get
    individually.

    Parameters
    ----------
    s1 : dimension of the input layer
    s2 : dimension of the first hidden layer.
    s3 : dimension of the output layer.

    Attributes
    ----------
    w1 : NN parameters for layer 1->2 (dimension s1 * s2)
    b1 : Bias term for layer 1->2 (dimension s2)
    w2 : NN parameters for layer 2->3 (dimension s2 * s3)
    b2 : Bias term for layer 2->3 (dimension s3)
    """
    def __init__(self, s1, s2, s3):
        self._s1 = s1
        self._s2 = s2
        self._s3 = s3
        self._start_idx = [0,
                           s1 * s2,
                           s1 * s2 + s2 * s3,
                           s1 * s2 + s2 * s3 + s2,
                           s1 * s2 + s2 * s3 + s2 + s3]

        self.arr1d = np.zeros(self._start_idx[-1])

    def rand_init(self):
        """Randomly initialize the matrix w1 and w2.
        The bias terms are set to 0.
        """
        rand_range = np.sqrt(6.0 / (self._s1 + self._s2 + 1))
        self.w1 = rand(self._s1, self._s2) * 2 * rand_range - rand_range
        self.w2 = rand(self._s2, self._s3) * 2 * rand_range - rand_range
        self.b1 = 0
        self.b2 = 0

    def copy(self):
        """Return a copy of the object."""
        copy = NetworkParams(self._s1, self._s2, self._s3)
        copy.arr1d = self.arr1d.copy()
        return copy

    @property
    def size(self):
        return self.arr1d.size

    @property
    def w1(self):
        return (
            self.arr1d[self._start_idx[0]:self._start_idx[1]].reshape(
                self._s1, self._s2))

    @w1.setter
    def w1(self, value):
        self.arr1d[self._start_idx[0]:self._start_idx[1]] = np.ravel(value)

    @property
    def w2(self):
        return (
            self.arr1d[self._start_idx[1]:self._start_idx[2]].reshape(
                self._s2, self._s3))

    @w2.setter
    def w2(self, value):
        self.arr1d[self._start_idx[1]:self._start_idx[2]] = np.ravel(value)

    @property
    def b1(self):
        return self.arr1d[self._start_idx[2]:self._start_idx[3]]

    @b1.setter
    def b1(self, value):
        self.arr1d[self._start_idx[2]:self._start_idx[3]] = value

    @property
    def b2(self):
        return self.arr1d[self._start_idx[3]:self._start_idx[4]]

    @b2.setter
    def b2(self, value):
        self.arr1d[self._start_idx[3]:self._start_idx[4]] = value


class SparseAutoencoder(object):
    """Sparse autoencoder that performs unsupervised learning to extract
    interesting features.

    Parameters
    ----------
    input_size : Size of input layer
    hidden_size : Size of hidden layer
    L : Regularization strength for weight decay term
    rho : Sparsity parameter
    beta: Sparsity penalty weight

    Attributes
    ----------
    L, rho, beta : See above
    params : A NetworkParams object storing the parameters for the
    neural network
    """
    def __init__(self, hidden_size=25, L=0, rho=0.1, beta=0):
        self._hidden_size = hidden_size
        self.L = L
        self.rho = rho
        self.beta = beta
        self.params = None
        self._params_grad = None
        self._output = []
        self._mean_activation = []

    def check_grad(self, input_data):
        """Compute the error between numerical and analytical gradient.

        Parameters
        ----------
        input_data : Input training data.

        Returns
        -------
        error : Error of the gradients. Defined as
        Norm(Gn - Ga) / Norm(Gn + Ga), where Gn and Ga are the analytical and
        numerical gradients.
        analytical_grad : analytical gradients (numpy array).
        numerical_grad : numerical gradients (numpy array).
        """
        input_data = self._input_convert(input_data)
        self._init_params(input_data)
        self._feed_forward(input_data)
        self._backprop(input_data)
        numerical_grad = self._get_numerical_grad(input_data).arr1d
        analytical_grad = self._params_grad.arr1d
        norm1 = np.sqrt(np.sum((numerical_grad - analytical_grad) ** 2))
        norm2 = np.sqrt(np.sum((numerical_grad + analytical_grad) ** 2))
        return norm1 / norm2, analytical_grad, numerical_grad

    def train(self, input_data):
        """Train parameters of the SparseAutoencoder with input data.

        Parameters
        ----------
        input_data : Input training data.
        """
        def min_func(params):
            """Function for minimization. Return the value and gradient of
            the cost function.
            """
            self.params.arr1d = params
            self._feed_forward(input_data)
            self._backprop(input_data)
            value = self._cost_func(input_data)
            grad = self._params_grad.arr1d
            return value, grad

        input_data = self._input_convert(input_data)
        self._init_params(input_data)
        init_guess = self.params.arr1d
        result = minimize(min_func, init_guess, method='L-BFGS-B', jac=True,
                          options={'maxiter': 400})
        self.params.arr1d = result.x

    def encode(self, input_data):
        """Encode the data using the trained parameter.

        Parameters
        ----------
        input_data : Input data to be encoded.

        Returns
        -------
        sparse_code : The output of hidden unit, which is a sparse
        representation of the data.
        auto_code : The output layer, which should have the same dimension as
        and look similar to the input.
        """
        if self.params is None:
            raise RuntimeError("The model need to be trained first "
                               "by calling the train() method")
        input_data = self._input_convert(input_data)
        if self._input_size != input_data.shape[1]:
            raise ValueError("Input data is not the same "
                             "length as training data")
        self._feed_forward(input_data)
        return self._output[0], self._output[1]

    def _init_params(self, input_data):
        input_size = input_data.shape[1]
        self._input_size = input_size
        self.params = NetworkParams(input_size, self._hidden_size, input_size)
        self._params_grad = NetworkParams(input_size, self._hidden_size,
                                          input_size)
        self.params.rand_init()

    def _cost_func(self, input_data):
        """Target cost function."""
        return (
            0.5 * np.mean(np.sum((input_data - self._output[1]) ** 2,
                          axis=1)) +
            0.5 * self.L * (np.sum(self.params.w1 ** 2) +
                            np.sum(self.params.w2 ** 2)) +
            np.sum(self._kl_divergence) * self.beta)

    def _feed_forward(self, input_data):
        output = []
        z0 = input_data.dot(self.params.w1) + self.params.b1
        output.append(self._logistic(z0))
        z1 = output[0].dot(self.params.w2) + self.params.b2
        output.append(self._logistic(z1))
        self._output = output
        self._mean_activation = np.mean(output[0], axis=0)

    def _backprop(self, input_data):
        # The expression below uses the property of logistic function, that
        # f'(x) = f(x) * (1 - f(x))
        delta1 = (-(input_data - self._output[1]) *
                  self._output[1] * (1 - self._output[1]))
        delta0 = (
            (delta1.dot(self.params.w2.T) +
             self.beta * self._kl_factor[None, :]) *
            self._output[0] * (1 - self._output[0]))

        self._params_grad.w1 = (
            self.L * self.params.w1 +
            input_data.T.dot(delta0) / input_data.shape[0])
        self._params_grad.w2 = (
            self.L * self.params.w2 +
            self._output[0].T.dot(delta1) / input_data.shape[0])
        self._params_grad.b1 = np.mean(delta0, axis=0)
        self._params_grad.b2 = np.mean(delta1, axis=0)

    def _get_numerical_grad(self, input_data):
        """Compute numerical gradients."""
        epsilon = 1e-4
        numerical_grad = self._params_grad.copy()
        for i in xrange(numerical_grad.size):
            self.params.arr1d[i] += epsilon
            self._feed_forward(input_data)
            value_plus = self._cost_func(input_data)

            self.params.arr1d[i] -= 2 * epsilon
            self._feed_forward(input_data)
            value_minus = self._cost_func(input_data)

            self.params.arr1d[i] += epsilon
            numerical_grad.arr1d[i] = (
                (value_plus - value_minus) / epsilon * 0.5)
        return numerical_grad

    @staticmethod
    def _logistic(x):
        """The logistic activation function."""
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _input_convert(input_data):
        """Convert input data into a proper 2D array for analysis."""
        input_data = np.asarray(input_data)
        if input_data.ndim > 2:
            raise ValueError("Input data is not 2D")

        if input_data.ndim == 1:
            # Convert N-element 1D array to a (1, N) 2D array
            input_data = input_data.reshape(1, input_data.size)
        return input_data

    @property
    def _kl_divergence(self):
        rho = self.rho
        rho_cap = self._mean_activation
        return (rho * np.log(rho / rho_cap) +
                (1.0 - rho) * np.log((1.0 - rho) / (1.0 - rho_cap)))

    @property
    def _kl_factor(self):
        rho = self.rho
        rho_cap = self._mean_activation
        return (1.0 - rho) / (1.0 - rho_cap) - rho / rho_cap
