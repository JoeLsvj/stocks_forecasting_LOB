import matplotlib.pyplot as plt

# Plotting function
def plot_history(history, title, ylabel):
    epochs = range(1, len(history) + 1)
    plt.plot(epochs, history, label = ylabel)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_2_histories(history, history2, title, label_plot1, label_plot2):
    epochs = range(1, len(history) + 1)
    plt.plot(epochs, history, label=label_plot1)
    plt.plot(epochs, history2, label=label_plot2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3_histories(history1, history2, history3, title, label_plot1, label_plot2, label_plot3):
    epochs = range(1, len(history1) + 1)
    plt.plot(epochs, history1, label=label_plot1)
    plt.plot(epochs, history2, label=label_plot2)
    plt.plot(epochs, history3, label=label_plot3)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_4_histories(history1, history2, history3, history4, title, label_plot1, label_plot2, label_plot3, label_plot4):
    epochs = range(1, len(history1) + 1)
    plt.plot(epochs, history1, label=label_plot1)
    plt.plot(epochs, history2, label=label_plot2)
    plt.plot(epochs, history3, label=label_plot3)
    plt.plot(epochs, history4, label=label_plot4)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_5_histories(history1, history2, history3, history4, history5, title, labels):
    epochs = range(1, len(history1) + 1)
    plt.plot(epochs, history1, label=labels[0])
    plt.plot(epochs, history2, label=labels[1])
    plt.plot(epochs, history3, label=labels[2])
    plt.plot(epochs, history4, label=labels[3])
    plt.plot(epochs, history5, label=labels[4])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
