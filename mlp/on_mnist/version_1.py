import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import sys

latent_size = 100
image_size = 28 * 28

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        hidden_size = 1000

        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        hidden_size = 1000
        output_size = 1
        self.main = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
    
path = sys.argv[1]

def print_graphic(data):
    for i in range(0, len(data)):
        row = data[i]
        for j in range(0, len(row)):
            if row[j] > 127:
                print("#", end="")
            else:
                print("-", end="")
        print("")

generator = Generator()
discriminator = Discriminator()
dataset_elements = []
try:
    with open(path, "rb") as file:

        verification = int.from_bytes(file.read(2), byteorder="big")
        if verification != 0:
            print("Note: Invalid verification. Should always be 0.")
        
        type = int.from_bytes(file.read(1), byteorder="big")
        unsigned_byte_case = 8
        if type != unsigned_byte_case:
            print("Note: This implementation only idx with unsigned byte type.")

        dim = int.from_bytes(file.read(1), byteorder="big")
        if dim != 3:
            print("Note: This implementation only supports 3d idx.")
        
        size = []
        total_size = 1
        for i in range(0, dim):
            dim_size = int.from_bytes(file.read(4), byteorder="big")
            size.append(dim_size)
            total_size *= dim_size

        print("Data size: ")
        print(size)

        logging_i = 0
        logging_examples = 100
        logging_frequency = size[0] // logging_examples
        for i in range(0, size[0]):
            element = []
            for j in range(0, size[1]):
                row = []
                for k in range(0, size[2]):
                    byte_int = int.from_bytes(file.read(1), byteorder="big")
                    row.append(byte_int)
                element.append(row)
            
            if logging_i % logging_frequency == 0:
                    print(f"Example element (element {i}):")
                    print_graphic(element)
            logging_i += 1
            dataset_elements.append(element)

            flat_element = [item for sublist in element for item in sublist]
            tensor = torch.Tensor(flat_element)
            discriminator_result = discriminator(tensor)
            #print("Discriminator result: {}".format(discriminator_result))
        
        print("Data read complete.")
        print("Elements in data: " + str(len(dataset_elements)))
            
except Exception as e:
    print(e)


batch_size = 128
num_epochs = 64

class MNISTDataSet(torch.utils.data.Dataset):
    def __init__(self, path):
        pass

    def __len__(self):
        return len(dataset_elements)
    
    def __getitem__(self, idx):
        element = dataset_elements[idx]
        flattened = [item for sublist in element for item in sublist]
        tensor = torch.Tensor(flattened)
        return tensor


dataset = MNISTDataSet(path)
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=0
                                    )


optim_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002)
optim_generator = optim.Adam(generator.parameters(), lr=0.0002)

loss = nn.BCELoss()

# Training loop
for epoch in range(0, num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    for i, data in enumerate(loader, 0):
        discriminator_result = discriminator(data)
        
        # Generate fake batch.
        current_batch_size = len(data)
        latent_vector = torch.randn(current_batch_size, latent_size)
        generator_result = generator(latent_vector)
        # ------------------------------------------------------------

        label_for_fake = torch.zeros(current_batch_size, 1)
        label_for_real = torch.ones(current_batch_size, 1)
        
        error_real = loss(discriminator_result, label_for_real)
        error_real_logging_value = error_real.item()
        error_real.backward()

        error_fake = loss(discriminator(generator_result.detach()), label_for_fake)
        error_fake_logging_value = error_fake.item()
        error_fake.backward()

        error_discriminator = error_real + error_fake
        error_discriminator_logging_value = error_discriminator.item()
        optim_discriminator.step()

        # Train the generator
        error_generator = loss(discriminator(generator_result), label_for_real)
        error_generator_logging_value = error_generator.item()
        error_generator.backward()
        optim_generator.step()

        logging_period = 2
        if i % logging_period == 0 or i == len(loader) - 1:
            print("- Batch {}/{}".format(i, len(loader)))
            print("-- Error discriminator real: " + str(error_real_logging_value))
            print("-- Error discriminator fake: " + str(error_fake_logging_value))
            print("-- Error discriminator combined: " + str(error_discriminator_logging_value))
            print("-- Error generator: " + str(error_generator_logging_value))

