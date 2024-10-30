import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import os
from seaborn import violinplot, stripplot, jointplot, barplot
from math import sqrt
from scipy.stats import t
from icecream import ic

def create_st_parity_plot(real, predicted, figure_name, save_path=None):
    """
    Create a parity plot and display R2, MAE, and RMSE metrics.

    Args:
        real (numpy.ndarray): An array of real (actual) values.
        predicted (numpy.ndarray): An array of predicted values.
        save_path (str, optional): The path where the plot should be saved. If None, the plot is not saved.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
        matplotlib.axes._axes.Axes: The Matplotlib axes object.
    """
    # Calculate R2, MAE, and RMSE
    r2 = r2_score(real, predicted)
    mae = mean_absolute_error(real, predicted)
    rmse = np.sqrt(mean_squared_error(real, predicted))
    
    # Create the parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(real, predicted, alpha=0.7)
    plt.plot([min(real), max(real)], [min(real), max(real)], color='red', linestyle='--')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    
    # Display R2, MAE, and RMSE as text on the plot
    textstr = f'$R^2$ = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
    plt.gcf().text(0.15, 0.75, textstr, fontsize=12)
    
    # Save the plot if save_path is provided
    if save_path:
        # Ensure the directory exists
        save_path = os.path.join(save_path, figure_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    plt.close()

    

def create_it_parity_plot(real, predicted, index, figure_name, save_path=None):
    r2 = round(r2_score(real, predicted), 3)
    mae = round(mean_absolute_error(real, predicted), 3)
    rmse = round(np.sqrt(mean_squared_error(real, predicted)), 3)


    df = pd.DataFrame({'Real':real,
                       'Predicted': predicted,
                       'Idx': index})

    # Create a scatter plot
    fig = px.scatter(df, x='Real', y='Predicted', text = 'Idx', labels={'x': 'Real Values', 'y': 'Predicted Values'}, hover_data=['Idx', 'Real', 'Predicted'])
    fig.add_trace(go.Scatter(x=real, y=real, mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')))

    # Customize the layout
    fig.update_layout(
        title=f'Parity Plot',
        showlegend=True,
        legend=dict(x=0, y=1),
        xaxis=dict(showgrid=True, showline=True, zeroline=True, linewidth=1, linecolor='black'),
        yaxis=dict(showgrid=True, showline=True, zeroline=True, linewidth=1, linecolor='black'),
        plot_bgcolor='white',  # Set background color to white
    )

    # Display R2, MAE, and RMSE as annotations on the plot
    text_annotation = f'R2 = {r2:.3f}<br>MAE = {mae:.3f}<br>RMSE = {rmse:.3f}'
    fig.add_annotation(
        text=text_annotation,
        xref="paper", yref="paper",
        x=0.15, y=0.75,
        showarrow=False,
        font=dict(size=12),
    )

    # Save the plot as an HTML file if save_path is provided
    if save_path:
        # Ensure the directory exists
        save_path = os.path.join(save_path, figure_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)


def plot_tsne_with_subsets(data_df, feature_columns, color_column, set_column, fig_name=None, save_path=None, perplexity=30, learning_rate=200, n_iter=1000, show = False):
    # Perform t-SNE
    features = data_df[feature_columns]
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Add t-SNE results back to the DataFrame
    data_df['tSNE1'] = tsne_results[:, 0]
    data_df['tSNE2'] = tsne_results[:, 1]

    # Define subsets
    subsets = data_df[set_column].unique()

    # Create subplots
    fig, axes = plt.subplots(1, len(subsets), figsize=(20, 8), sharex=True, sharey=True, dpi=300)

    # Plot each subset
    for i, subset in enumerate(subsets):
        subset_df = data_df[data_df[set_column] == subset]
        scatter = axes[i].scatter(subset_df['tSNE1'], subset_df['tSNE2'], c=subset_df[color_column], cmap='plasma', s=100, alpha=0.7, edgecolors='w', linewidth=0.5)
        axes[i].set_title(f'{subset.capitalize()} Set', fontsize=18, pad=15)
        if i == 0:
            axes[i].set_ylabel('tSNE2', fontsize=20, labelpad=15)
        if i == 1:
            axes[i].set_xlabel('tSNE1', fontsize=20, labelpad=15)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].tick_params(axis='both', which='major', labelsize=12)

    # Add color bar to the right
    cbar = fig.colorbar(scatter, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('$\Delta \Delta G$', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Enhancing the overall look
    plt.suptitle('t-SNE 2D Visualization by Set', fontsize=22, y=1.05)
    sns.despine()
    #plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust layout to make room for color bar
    if save_path and fig_name:
        # Ensure the directory exists
        save_path = os.path.join(save_path, fig_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()

def create_training_plot(df, save_path):

    df = pd.read_csv(df)

    epochs = df.iloc[:,0]
    train_loss = df.iloc[:,1]
    val_loss = df.iloc[:,2]
    test_loss = df.iloc[:,3]

    min_val_loss_epoch = epochs[val_loss.idxmin()]

    # Create a Matplotlib figure and axis
    plt.figure(figsize=(10, 6), dpi=300)  # Adjust the figure size as needed
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o', linestyle='-')
    plt.plot(epochs, test_loss, label='Test Loss', marker='o', linestyle='-')

    plt.axvline(x=min_val_loss_epoch, color='gray', linestyle='--', label=f'Min Validation Epoch ({min_val_loss_epoch})')

    # Customize the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(False)
    plt.legend()

    # Save the plot in high resolution (adjust file format as needed)
    plt.savefig('{}/loss_vs_epochs.png'.format(save_path), bbox_inches='tight')

    plt.close()


def create_bar_plot(means:tuple, stds:tuple, min:float, max:float, metric:str, save_path:str, tml_algorithm:str):

    bar_width = 0.35

    mean_gnn, mean_tml = means
    std_gnn, std_tml = stds

    folds = list(range(1, 11))
    index = np.arange(10)

    plt.bar(index, mean_gnn, bar_width, label='GNN Approach', yerr=std_gnn, capsize=5)
    plt.bar(index+bar_width, mean_tml, bar_width, label=f'{tml_algorithm.upper()} Approach', yerr=std_tml, capsize=5)

    plt.ylim(min*.99, max *1.01)
    plt.xlabel('Fold Used as Test Set', fontsize = 16)

    label = 'Mean $R^2$ Value' if metric == 'R2' else f'Mean {metric} Value'
    plt.ylabel(label, fontsize = 16)

    plt.xticks(index + bar_width / 2, list(folds))

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.savefig(os.path.join(save_path, f'{metric}_GNN_vs_TML'), dpi=300, bbox_inches='tight')

    print('Plot {}_GNN_vs_TML has been saved in the directory {}'.format(metric,save_path))

    plt.clf()

    results_df = pd.DataFrame({
        'Fold': folds, 
        'Mean GNN': mean_gnn,
        'Std GNN': std_gnn,
        'MeanTML': mean_tml,
        'Std TML': std_tml,
        'p-value': np.nan,
        'Significance': np.nan
        })

    # Calculate t-statistic and p-value for each fold
    for i in range(len(folds)):
        se_gnn = std_gnn[i] / np.sqrt(len(folds))
        se_tml = std_tml[i] / np.sqrt(len(folds))
        se_diff = np.sqrt(se_gnn**2 + se_tml**2)
        t_stat = (mean_gnn[i] - mean_tml[i]) / se_diff
        df = (se_gnn**2 + se_tml**2)**2 / (se_gnn**4 / (len(folds) - 1) + se_tml**4 / (len(folds) - 1))
        p_value = 2 * t.sf(np.abs(t_stat), df)
        significant_diff = 'Yes' if p_value < 0.05 else 'No'
        
        results_df.at[i, 'p-value'] = p_value
        results_df.at[i, 'Significant Difference'] = significant_diff

    # Save the results to a CSV file
    results_df.to_csv(os.path.join(save_path, f'{metric}_GNN_vs_TML.csv'), index=False)
    print(f'CSV for {metric}_GNN_vs_TML has been saved in the directory {save_path}')


def create_violin_plot(data, save_path:str):

    violinplot(data = data, x='Test_Fold', y='Error', hue='Method', split=True, gap=.1, inner="quart", fill=False)

    plt.xlabel('Fold Used as Test Set', fontsize=18)
    plt.ylabel('$\Delta \Delta G_{real}-\Delta \Delta G_{predicted}$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax= plt.gca()
    ax.get_legend().remove()

    plt.savefig(os.path.join(save_path, f'Error_distribution_GNN_vs_TML_violin_plot'), dpi=300, bbox_inches='tight')
    plt.close()


def create_strip_plot(data, save_path:str):

    stripplot(data = data, x='Test_Fold', y='Error', hue='Method', size=3,  dodge=True, jitter=True, marker='D', alpha=.3)

    plt.xlabel('Fold Used as Test Set', fontsize=18)
    plt.ylabel('$\Delta \Delta G_{real}-\Delta \Delta G_{predicted}$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax= plt.gca()
    ax.get_legend().remove()

    plt.savefig(os.path.join(save_path, f'Error_distribution_GNN_vs_TML_strip_plot'), dpi=300, bbox_inches='tight')
    plt.close()

def create_parity_plot(data: pd.DataFrame, save_path:str, tml_algorithm:str):

    data['real_ddG'] = pd.to_numeric(data['real_ddG'], errors='coerce')
    data['predicted_ddG'] = pd.to_numeric(data['predicted_ddG'], errors='coerce')

    results_gnn = data[data['Method'] == 'GNN']

    g = jointplot(x="real_ddG", y="predicted_ddG", data=results_gnn,
                  kind="reg", truncate=False,
                  xlim=(-16.5, 16.5), ylim=(-16.5, 16.5),
                  color="#1f77b4", height=7,
                  scatter_kws={"s": 5, "alpha": 0.3})
    plt.axvline(x=0, color='black', linestyle='--', linewidth=.5)

    # add horizontal line at y=50
    plt.axhline(y=0, color='black', linestyle='--', linewidth=.5)

    plt.text(x=-7.5, y=15, s=f"False Positive", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')
    plt.text(x=7.5, y=15, s=f"True Positive", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    plt.text(x=-7.5, y=-.5, s=f"True Negative", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')
    plt.text(x=7.5, y=-.5, s=f"False Negative", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    g.ax_joint.xaxis.label.set_size(20)
    g.ax_joint.yaxis.label.set_size(20)

    g.ax_joint.set_xlabel('Real $\Delta \Delta G$ / kJ mol$^{-1}$')
    g.ax_joint.set_ylabel('Predicted $\Delta \Delta G$ / kJ mol$^{-1}$')

    g.ax_joint.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(os.path.join(save_path, f'parity_plot_GNN'), dpi=300, bbox_inches='tight')
    plt.close()
    

    results_tml = data[data['Method'] == tml_algorithm]

    g = jointplot(x="real_ddG", y="predicted_ddG", data=results_tml,
                  kind="reg", truncate=False,
                  xlim=(-16.5, 16.5), ylim=(-16.5, 16.5),
                  color="#ff7f0e", height=7,
                  scatter_kws={"s": 5, "alpha": 0.3})
    plt.axvline(x=0, color='black', linestyle='--', linewidth=.5)

    # add horizontal line at y=50
    plt.axhline(y=0, color='black', linestyle='--', linewidth=.5)

    plt.text(x=-7.5, y=15, s=f"False Positive", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')
    plt.text(x=7.5, y=15, s=f"True Positive", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    plt.text(x=-7.5, y=-.5, s=f"True Negative", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')
    plt.text(x=7.5, y=-.5, s=f"False Negative", fontsize=15, horizontalalignment='center', verticalalignment='center', color='black')

    g.ax_joint.xaxis.label.set_size(20)
    g.ax_joint.yaxis.label.set_size(20)

    g.ax_joint.set_xlabel('Real $\Delta \Delta G$ / kJ mol$^{-1}$')
    g.ax_joint.set_ylabel('Predicted $\Delta \Delta G$ / kJ mol$^{-1}$')

    g.ax_joint.tick_params(axis='both', which='major', labelsize=15)

    plt.savefig(os.path.join(save_path, f'parity_plot_{tml_algorithm}'), dpi=300, bbox_inches='tight')
    plt.close()



def plot_importances(df, save_path: str=None):
    plt.figure(figsize=(10, 6))

    ax = barplot(df, x="score", y="labels", estimator="sum", errorbar=None)
    ax.bar_label(ax.containers[0], fontsize=10)
    # ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center', horizontalalignment='right')

    plt.xlabel('Feature Importance Score', fontsize=16)
    plt.ylabel('Feature', fontsize=16)

    if save_path:
        # Save the figure before displaying it
        plt.savefig(os.path.join(save_path, 'node_feature_importance_plot'), dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

    print('Node feature importance plot has been saved in the directory {}'.format(save_path))
    plt.close()



def plot_mean_predictions(df, save_path: str=None, legend = True):

# Create the parity plot
    plt.figure(figsize=(12, 10))
    sns.set(style="whitegrid")

    # Scatter plot with hue for different methods
    scatter = sns.scatterplot(x='real_ddG', y='mean_predicted_ddG', data=df, s=100, edgecolor='k', hue='Method', palette='deep')

    # Add regression lines for each method and calculate metrics
    metrics_text = []
    for method in df['Method'].unique():
        subset = df[df['Method'] == method]
        sns.regplot(x='real_ddG', y='mean_predicted_ddG', data=subset, scatter=False, ci=None, label=f'Regression {method}', line_kws={'linestyle': '--'})
        
        # Calculate R2 and MAE
        r2 = r2_score(subset['real_ddG'], subset['mean_predicted_ddG'])
        mae = mean_absolute_error(subset['real_ddG'], subset['mean_predicted_ddG'])
        rmse = sqrt(mean_squared_error(subset['real_ddG'], subset['mean_predicted_ddG']))
        metrics_text.append(f"{method}: $R^2$: {r2:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Line of equality
    max_val = max(df['real_ddG'].max(), df['mean_predicted_ddG'].max())
    min_val = min(df['real_ddG'].min(), df['mean_predicted_ddG'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2, label='Line of Equality')

    # Titles and labels
    plt.xlabel('Real ΔΔG$^{\u2021}$ / kJ $mol^{-1}$', fontsize=26)
    plt.ylabel('Mean Predicted ΔΔG$^{\u2021}$ / kJ $mol^{-1}$', fontsize=26)

    # Enhancing the overall look
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    sns.despine(trim=True)

    # Add metrics as text
    metrics_text_str = "\n".join(metrics_text)
    plt.text(0.25, 0.1, metrics_text_str, ha='left', va='top', transform=plt.gca().transAxes, fontsize=16, bbox=dict(facecolor='white', alpha=0.8))

    # Adjust legend
    if legend:
        plt.legend(fontsize=16, title_fontsize=18)

    # Show the plot
    plt.tight_layout()

    if save_path:
        # Save the figure before displaying it
        plt.savefig(os.path.join(save_path, 'mean_predictions_plot'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    

def plot_distribution(df):
    # Use seaborn style for the plot
    sns.set(style="whitegrid")

    # Create the figure and subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    # First subplot: Histogram of reactions['%top']
    axes[0].hist(df['%top'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    #axes[0].set_title('Distribution of Top Facial Additions', fontsize=16)
    axes[0].set_xlabel('Reaction Top Addition (%)', fontsize=18)
    axes[0].set_ylabel('Frequency', fontsize=18)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='both', which='major', labelsize=18)  # Increased label size
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Second subplot: Histogram of reactions['ddG']
    axes[1].hist(df['ddG'], bins=10, color='lightcoral', edgecolor='black', alpha=0.7)
    #axes[1].set_title('Distribution of $\Delta \Delta$G', fontsize=16)
    axes[1].set_xlabel('$\Delta \Delta$G (kJ/mol)', fontsize=18)
    #axes[1].set_ylabel('Frequency', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='both', which='major', labelsize=18)  # Increased label size
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Adjust layout to make sure everything fits
    plt.tight_layout()

    # Show the plot
    plt.show()


def parity_mean(df, save_path: str=None):

# Create the parity plot
    plt.figure(figsize=(12, 10))
    sns.set(style="whitegrid")

    # Scatter plot with hue for different methods
    scatter = sns.scatterplot(x='real_ddG', y='mean_predicted_ddG', data=df, s=100, edgecolor='k', palette='deep')

    # Add regression lines for each method and calculate metrics
    metrics_text = []

    sns.regplot(x='real_ddG', y='mean_predicted_ddG', data=df, scatter=False, ci=None, label=f'Regression {df}', line_kws={'linestyle': '--'})
    
    # Calculate R2 and MAE
    r2 = r2_score(df['real_ddG'], df['mean_predicted_ddG'])
    mae = mean_absolute_error(df['real_ddG'], df['mean_predicted_ddG'])
    rmse = sqrt(mean_squared_error(df['real_ddG'], df['mean_predicted_ddG']))
    metrics_text.append(f"$R^2$: {r2:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Line of equality
    max_val = max(df['real_ddG'].max(), df['mean_predicted_ddG'].max())
    min_val = min(df['real_ddG'].min(), df['mean_predicted_ddG'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2, label='Line of Equality')

    # Titles and labels
    plt.xlabel('Real ΔΔG$^{\u2021}$ / kJ $mol^{-1}$', fontsize=32)
    plt.ylabel('Mean Predicted ΔΔG$^{\u2021}$ / kJ $mol^{-1}$', fontsize=32)

    # Enhancing the overall look
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(True, linestyle='--', alpha=0.7)
    sns.despine(trim=True)

    # Add metrics as text
    metrics_text_str = "\n".join(metrics_text)
    plt.text(0.25, 0.1, metrics_text_str, ha='left', va='top', transform=plt.gca().transAxes, fontsize=30, bbox=dict(facecolor='white', alpha=0.8))

    # Adjust legend
    #plt.legend(fontsize=16, title_fontsize=18)

    # Show the plot
    plt.tight_layout()

    if save_path:
        # Save the figure before displaying it
        plt.savefig(os.path.join(save_path, 'mean_predictions_plot'), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_error_distribution(df, save_path: str=None):
    # Calculate residuals
    df['residuals'] = df['mean_predicted_ddG'] - df['real_ddG']

    # Create the error distribution plot
    plt.figure(figsize=(12, 10))
    sns.set(style="whitegrid")
    sns.histplot(df['residuals'], kde=True, edgecolor='k', color='blue')
    plt.xlabel('Residuals (Predicted - Real) ΔΔG$^{\u2021}$ / kJ $mol^{-1}$', fontsize=32)
    plt.ylabel('Frequency', fontsize=32)
    #plt.title('Error Distribution', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
    sns.despine(trim=True)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.tight_layout()

    if save_path:
        # Save the figure
        plt.savefig(os.path.join(save_path, 'error_distribution_plot.png'), dpi=300, bbox_inches='tight')
    
    plt.close()