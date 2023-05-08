# Generative Models

This repository is dedicated to sharing open-source implementations of fundamental generative models in artificial general intelligence (AGI). The goal is to provide a comprehensive resource for researchers and practitioners interested in exploring and experimenting with these models.

## Models Included

Currently, this repository includes the following generative models:

- Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)
- Autoregressive models
- Normalizing Flows
- Boltzmann Machines
- Hopfield Networks
- Diffusion Model

Each model has a separate directory containing the implementation code and a brief description of the model.

## Usage

The implementations are provided in Python using PyTorch. To use these models, clone this repository and install the required dependencies specified in the `requirements.txt` file. Each model has its own script for training and generating samples. The script can be run using the command `python <model_name>_train.py` and `python <model_name>_generate.py`.

## Contributions

Contributions are welcome in the form of new models, bug fixes, or improved implementations. If you wish to contribute, please follow the guidelines provided in the `CONTRIBUTING.md` file.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.

## References

The implementations in this repository are based on the following papers:

- [Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.](https://arxiv.org/abs/1312.6114)
- [Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).](https://arxiv.org/abs/1406.2661)
- [Oord, A. van den, Kalchbrenner, N., & Kavukcuoglu, K. (2016). Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759.](https://arxiv.org/abs/1601.06759)
- [Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density estimation using real NVP. arXiv preprint arXiv:1605.08803.](https://arxiv.org/abs/1605.08803)
- [Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural computation, 14(8), 1771-1800.](https://www.cs.toronto.edu/~hinton/absps/hinton_techreport.pdf)
- [Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8), 2554-2558.](https://www.pnas.org/content/79/8/2554)
