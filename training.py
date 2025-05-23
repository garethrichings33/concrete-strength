from numpy import float32
import torch
from torch import tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split
from raw_data_handler import extract_raw_data


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(8, 20)
        self.linear_out = nn.Linear(20, 1)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        return self.linear_out(x)


def extract_features_responses(dataframe, no_features):
    features = dataframe.iloc[:, 0:no_features].to_numpy()
    responses = dataframe.iloc[:, no_features].to_numpy()

    return features, responses


def create_dataset(features, responses):

    features_tensor = tensor(features.astype(float32))
    responses_tensor = tensor(responses.astype(float32))

    return TensorDataset(features_tensor, responses_tensor)


def train_one_epoch(model, training_dataloader, optimiser, criterion):
    """
    Function to train a single epoch

    Returns the mean running loss
    """
    model.train()
    running_loss = 0.
    for data in training_dataloader:
        features, responses = data
        optimiser.zero_grad()
        predictions = model(features)
        loss = criterion(predictions.squeeze(), responses)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()

    return running_loss/len(training_dataloader.dataset)


def get_validation_loss(model, dataloader, criterion):
    pass


def train_model(training_dataloader, validation_dataloader):
    model = Network()
    criterion = nn.MSELoss(reduction="sum")
    optimiser = torch.optim.SGD(model.parameters(),
                                lr=1.e-8,
                                weight_decay=0.,
                                momentum=0.)

    EPOCHS = 1_001
    loss_tracker = []
    min_training_loss = 1.e9
    min_validation_loss = 1.e9

    for epoch in range(EPOCHS):
        training_loss = train_one_epoch(model,
                                        training_dataloader,
                                        optimiser,
                                        criterion)
        validation_loss = get_validation_loss(model,
                                              validation_dataloader,
                                              criterion)

        if training_loss < min_training_loss:
            min_training_loss = training_loss
            min_training_loss_epoch = epoch

        if validation_loss < min_validation_loss:
            min_validation_loss = training_loss
            min_validation_loss_epoch = epoch

        print(f"Epoch: {epoch}, Training Loss: {training_loss:14.12f}, "
              f"Validation Loss: {validation_loss:14.12f}")

        if epoch % 50 == 0:
            loss_tracker.append((epoch,
                                 training_loss,
                                 validation_loss))

    print(f"Minimum Training Loss: {min_training_loss:14.12f} "
          f"at epoch {min_training_loss_epoch}")
    print(f"Minimum Validation Loss: {min_validation_loss:14.12f} "
          f"at epoch {min_validation_loss_epoch}")


if __name__ == "__main__":

    # Use deterministic algorithms for testing.
    seed_everything(12)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Get raw data from the CSV file
    path = "concrete+compressive+strength/"
    filename = "Concrete_Data.csv"
    raw_dataframe = extract_raw_data(filename, path)

    # Split data into features array and response array
    no_features = 8
    data_features, data_responses = extract_features_responses(
        raw_dataframe, no_features)

    # Split data into training and validation sets
    # Use a specified random_state for repeatability during testing
    (training_features,
     validation_features,
     training_responses,
     validation_responses) = train_test_split(data_features,
                                              data_responses,
                                              test_size=0.2,
                                              random_state=1)

    # Create DataSets
    training_dataset = create_dataset(training_features, training_responses)
    validation_dataset = create_dataset(
        validation_features, validation_responses)

    # Create training and validation DataLoaders
    batch_size = 20
    training_dataloader = DataLoader(training_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=batch_size,
                                       shuffle=False)

    #  Train network
    train_model(training_dataloader, validation_dataloader)
