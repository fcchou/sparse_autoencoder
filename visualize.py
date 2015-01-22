import matplotlib.pyplot as plt
from sparse_autoencoder import (
    SparseAutoencoder, sample_training_images, PATCH_SIZE)

training_data = sample_training_images()

encoder = SparseAutoencoder(hidden_size=25, rho=0.01, L=0.0001, beta=3)
encoder.train(training_data)

w1 = encoder.params.w1

fig = plt.figure()
for i in xrange(w1.shape[-1]):
    image = w1[:, i].reshape(*PATCH_SIZE)
    ax = fig.add_subplot(5, 5, i+1)
    ax.imshow(image, cmap='binary')
    plt.axis('off')
plt.show()
