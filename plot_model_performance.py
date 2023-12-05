import matplotlib.pyplot as plt
import os

def plot_model_performance(model_metrics, output_dir='output'):
    """
    Saves a scatter plot of model performance metrics to a file.
    
    This function takes a list of tuples containing model performance metrics,
    creates a scatter plot visualizing the precision and recall for each model,
    and saves it to the specified output directory.

    Parameters:
    model_metrics (list of tuples): A list where each tuple contains the model name,
                                    accuracy, precision, recall, and F1 score.
    output_dir (str): The directory where the output image will be saved. Defaults to 'output'.

    The function does not return anything.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'model_performance_comparison.png')

    fig, ax = plt.subplots()
    for model, acc, prec, rec, f1 in model_metrics:
        ax.scatter(prec, rec, label=f'{model} (Acc: {acc:.2f}, F1: {f1:.2f})')

    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_title('Model Performance Comparison')
    ax.legend()

    plt.savefig(output_file)
    plt.close(fig)


