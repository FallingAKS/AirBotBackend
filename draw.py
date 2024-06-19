import matplotlib.pyplot as plt
from test import train_losses, validate_Losses


def draw():
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='train losses', color='b', marker='o')
    plt.plot(validate_Losses, label='validate losses', color='r', marker='x')
    plt.title('Train and Validate Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    draw()
