import torch
import torch.nn as nn

# Przykładowe dane wejściowe - obraz o wymiarach 1x3x4x4 (batch_size x channels x height x width)
input_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0],
                              [13.0, 14.0, 15.0, 16.0]]]])

# Definicja warstwy MaxPool2d
maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

# Przejście przez warstwę MaxPool2d
output = maxpool_layer(input_data)

# Wyświetlenie danych wyjściowych
print("Dane wyjściowe po MaxPooling:")
print(output)
