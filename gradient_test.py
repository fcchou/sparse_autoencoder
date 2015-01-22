import numpy as np
import matplotlib.pyplot as plt
from sparse_autoencoder import SparseAutoencoder, sample_training_images

training_data = sample_training_images()[:100, :]
encoder = SparseAutoencoder(rho=0.01, L=0.0001, beta=3)
error, analytical_grad, numerical_grad = encoder.check_grad(training_data)
print "Gradient Error:", error

plt.plot(analytical_grad, numerical_grad, 'ro')
plt.plot([np.min(analytical_grad), np.max(analytical_grad)],
         [np.min(analytical_grad), np.max(analytical_grad)], 'k-')
plt.xlabel('Analytical')
plt.ylabel('Numerical')
plt.show()
