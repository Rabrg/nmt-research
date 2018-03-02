import torch.nn as nn

from translation.lang import *
from translation.masked_cross_entropy import *
from translation.time import *

networks = 3
hidden = 1024
num_epochs = 5
learning_rate = 0.001

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_lang_total, output_lang_total, pairs_total = prepare_data('cmn', 'eng', 'casict2015_total.txt', False)
input_lang_train, output_lang_train, pairs_train = prepare_data('cmn', 'eng', 'casict2015_train.txt', False)
input_lang_test, output_lang_test, pairs_test = prepare_data('cmn', 'eng', 'casict2015_test.txt', False)

net = Net(input_lang_total.n_words * networks, hidden, input_lang_total.n_words)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (input_sent, output_sent) in enumerate(pairs_train):
        # Convert torch tensor to Variable
        input_sent = Variable(input_sent.view(-1, 28 * 28))
        output_sent = Variable(output_sent)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(input_sent)
        loss = criterion(outputs, output_sent)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, i + 1, len(pairs_total), loss.data[0]))
            # Save the Model
            torch.save(net.state_dict(), 'fine_tune.pkl')

