import matplotlib.pyplot as plt

def plot_bar_chart(data, title, xlabel, ylabel, color, alpha=0.7, figsize=(10, 5)):
    """
    Generates a bar chart based on the provided data and parameters.

    Parameters:
    - data (dict): A dictionary with 'x' (labels) and 'y' (values) keys.
    - title (str): Title of the chart.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - color (str): Bar color.
    - alpha (float): Bar transparency.
    - figsize (tuple): Size of the figure.

    Returns:
    - None
    """
    plt.figure(figsize=figsize)
    plt.bar(data['x'], data['y'], alpha=alpha, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def visualize_model_data(models_data):
    """
    Visualizes loss and accuracy data for each model in the models_data dictionary.

    Parameters:
    - models_data (dict): A dictionary containing model names as keys and their
      respective data (ELO ranges, losses, and accuracies) as values.

    Returns:
    - None
    """
    for model_name, data in models_data.items():
        # Plot Loss
        plot_bar_chart(
            data={'x': data['elo_ranges'], 'y': data['losses']},
            title=f'Loss Across ELO Ranges ({model_name})',
            xlabel='ELO Range',
            ylabel='Loss',
            color='blue'
        )
        
        # Plot Accuracy
        plot_bar_chart(
            data={'x': data['elo_ranges'], 'y': data['accuracies']},
            title=f'Accuracy Across ELO Ranges ({model_name})',
            xlabel='ELO Range',
            ylabel='Accuracy (%)',
            color='green'
        )

# Main process for fine-tuning
if __name__ == "__main__":
    # Data for ELO ranges, losses, and accuracies for each model
    models_data = {
        "DNN": {
            "elo_ranges": ['<1000', '1000-1500', '1500-2000', '>2000'],
            "losses": [5.9137, 5.7089, 5.7176, 5.7024],
            "accuracies": [9.77, 9.99, 9.41, 8.83]
        },
        "CNN": {
            "elo_ranges": ['<1000', '1000-1500', '1500-2000', '>2000'],
            "losses": [5.0660, 5.0810, 5.1264, 5.1905],
            "accuracies": [10.77, 10.46, 10.24, 9.66]
        },
        "Fine-Tuned Transformer": {
            "elo_ranges": ['<1000', '1000-1500', '1500-2000', '>2000'],
            "losses": [0.4100, 0.4782, 0.5559, 0.6249],
            "accuracies": [91.81, 90.36, 88.88, 87.61]
        }
    }

    # Visualize the data for all models
    visualize_model_data(models_data)