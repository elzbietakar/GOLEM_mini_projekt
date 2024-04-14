
def train_epoch(model, criterion, optimizer, loader, id):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{id + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0