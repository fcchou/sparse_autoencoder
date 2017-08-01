Sparse Autoendcoder Excercise
=============================

This is an exercise on implementing a sparse autoencoder. The excerise
comes from Prof. Andrew Ng's [UFLDL tutorial](http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder)

The description and tutorial of the excerise can be also found in the PDF
files in this folder.

Note that I am implementing it in Python instead of MATLAB.
The basic algorithm is the same.

The code requires numpy and scipy. matplotlib is needed for visualization.

Example
-------
- Gradient checking: [gradient_test.py](../master/gradient_test.py)

![alt text](../master/gradient_check.png)

- Training and visualize hidden units: [visualize.py](../master/visualize.py)

![alt text](../master/hidden_units.png)

You may also play with the parameters in the code to see the difference on
the results.

The main codes are in [sparse_autoencoder.py](../master/sparse_autoencoder.py).
Here the SparseAutoencoder class is designed to be quite general, so you may use it for other types of data.
