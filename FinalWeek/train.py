# imports
from model import ConvNet
import torch.optim as optim
import torch.nn as nn

# training
def train(params):
    input_dim = params['input_dim']
    output_dim = params['output_dim']
    num_hidden_layers = params['num_hidden_layers']
    convolution_filters = params['convolution_filters']
    learning_rate = params['learning_rate']
    policy_weight = params['policy_weight']
    value_weight = params['value_weight']
    num_epochs = params['epochs']

    model = ConvNet(input_dim, output_dim, num_hidden_layers, convolution_filters)

    cross_entropy_loss = nn.CrossEntropyLoss()
    mean_squared_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = {
        'policy_head': cross_entropy_loss,
        'value_head': mean_squared_loss
    }

    loss_weights = {
        'policy_head': policy_weight,
        'value_head': value_weight
    }

    train_loader = {} # TODO: to be changed

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            data, targets = batch['data'], batch['targets']

            policy_out, value_out = model(data)

            policy_loss = losses['policy_head'](policy_out, targets['policy'])
            value_loss = losses['value_head'](value_out, targets['value'])
            loss = loss_weights['policy_head'] * policy_loss + loss_weights['value_head'] * value_loss

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
















