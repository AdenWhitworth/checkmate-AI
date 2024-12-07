import matplotlib.pyplot as plt

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

# Generate bar charts for each model
for model_name, data in models_data.items():
    # Loss bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(data['elo_ranges'], data['losses'], alpha=0.7, color='blue')
    plt.title(f'Loss Across ELO Ranges ({model_name})')
    plt.xlabel('ELO Range')
    plt.ylabel('Loss')
    plt.show()
    
    # Accuracy bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(data['elo_ranges'], data['accuracies'], alpha=0.7, color='green')
    plt.title(f'Accuracy Across ELO Ranges ({model_name})')
    plt.xlabel('ELO Range')
    plt.ylabel('Accuracy (%)')
    plt.show()
