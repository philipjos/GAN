# GAN

## GAN with MLP networks
### GAN working with audio samples
Usage: python mlp/on_audio_samples/main.py path/to/idx_file
Notes: There seems to be a high error on the discriminator fake in this experiment.

### GAN working with mnist images
#### Version 1
Usage: python mlp/on_mnist/version_1.py path/to/idx_file
Notes: Similar to the version working with audio samples, there seem to be a high error on the discriminator fake.

#### Version 2
Usage: python mlp/on_mnist/version_2.py path/to/idx_file
Version 2 adds a `zero_grad()` call on the networks in each iteration.
Notes: This version shows better results in terms of the discriminator fake error, but seems to give a high error on the generator.