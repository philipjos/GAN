# GAN

## GAN with MLP networks
### GAN working with audio samples
There seems to be a high error on the discriminator fake in this version.

### GAN working with mnist images
#### Version 1
Similar to the version working with audio samples, there seem to be a high error on the discriminator fake.

#### Version 2
Version 2 adds a `zero_grad()` call on the networks in each iteration.
This shows better results in terms of the discriminator fake error, but seems to give a high error on the generator.