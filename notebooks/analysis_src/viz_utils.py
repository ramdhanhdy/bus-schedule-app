import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def create_correlation_colormap():
    """Create a custom colormap for correlation visualization."""
    colors = ["#1a237e", "#FFFFFF", "#b71c1c"]  # Dark blue -> White -> Dark red
    return LinearSegmentedColormap.from_list("correlation_cmap", colors, N=256)

def style_correlation_plot(ax, title, fig=None):
    """Apply consistent styling to correlation plots."""
    ax.set_title(title, fontsize=14, pad=20, color='#1a237e', fontweight='bold')
    ax.set_xlabel('Features', fontsize=12, color='#1a237e')
    ax.set_ylabel('Correlation Coefficient', fontsize=12, color='#1a237e')
    
    ax.tick_params(axis='both', colors='#1a237e', labelsize=10)
    plt.xticks(rotation=45, ha='right')
    
    ax.grid(True, alpha=0.15, color='#1a237e', linestyle='--')
    ax.axhline(y=0, color='#b71c1c', linestyle='--', alpha=0.6, linewidth=1.5)
    
    for spine in ax.spines.values():
        spine.set_color('#1a237e')
        spine.set_linewidth(1.5)
    
    if fig:
        plt.tight_layout()

def add_correlation_legend(ax, transform=None):
    """Add correlation strength interpretation legend."""
    legend_text = ('Correlation Strength:\n'
                  '━━━ Strong: |r| > 0.5\n'
                  '━━━ Moderate: 0.3 < |r| < 0.5\n'
                  '━━━ Weak: |r| < 0.3')
    
    bbox_props = dict(
        facecolor='white',
        edgecolor='#1a237e',
        alpha=0.95,
        boxstyle='round,pad=0.5',
        linewidth=1.5
    )
    
    transform = transform or ax.transAxes
    ax.text(1.15, 0.95, legend_text,
            transform=transform,
            bbox=bbox_props,
            fontsize=10,
            color='#1a237e',
            verticalalignment='top')

def plot_weather_correlations(weather_df, merged_df):
    """Create an enhanced correlation plot for weather features."""
    excluded_cols = {
        'datetime', 'time', 'hour', 'hour_sin', 'hour_cos',
        'month_num', 'month_sin', 'month_cos', 'month',
        'humidity_comfort', 'heat_stress',
        'temp_rolling_std', 'temp_change', 'pressure_change',
        'visibility_cleaned', 'humidity_cleaned', 'wind_speed_cleaned'
    }
    
    weather_features = [col for col in weather_df.columns if col not in excluded_cols]
    
    correlations = merged_df[weather_features + ['duration']].corr()['duration']
    correlations = correlations[correlations.index != 'duration'].sort_values(ascending=False)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    cmap = create_correlation_colormap()
    norm = plt.Normalize(vmin=-1, vmax=1)
    
    bars = correlations.plot(kind='bar', ax=ax, color='gray', alpha=0.7)
    
    for bar, value in zip(bars.patches, correlations.values):
        bar.set_color(cmap(norm(value)))
        bar.set_alpha(0.8)
        
        label_color = '#1a237e' if abs(value) < 0.6 else 'white'
        y_pos = value + 0.01 if value >= 0 else value - 0.02
        
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{value:.2f}',
                ha='center', va='bottom' if value >= 0 else 'top',
                color=label_color,
                fontsize=9,
                fontweight='bold')
    
    style_correlation_plot(ax, 'Weather Feature Correlations with Trip Duration', fig)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Correlation Strength', fontsize=12, color='#1a237e')
    cbar.ax.tick_params(colors='#1a237e')
    
    add_correlation_legend(ax)
    
    return fig, ax

def plot_correlation_matrix(df, target_col=None, figsize=(14, 10)):
    """
    Create an enhanced correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing features for correlation analysis
    target_col : str, optional
        If provided, will highlight correlations with this target column
    figsize : tuple, default=(10, 8)
        Figure size for the plot
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    corr_matrix = numeric_df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(corr_matrix,
                mask=mask,
                cmap=create_correlation_colormap(),
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .5},
                annot=True,
                fmt='.2f',
                ax=ax)
    
    # Highlight target column if specified
    if target_col and target_col in numeric_df.columns:
        target_idx = list(numeric_df.columns).index(target_col)
        ax.axhline(y=target_idx, color='#E74C3C', alpha=0.3)
        ax.axvline(x=target_idx, color='#E74C3C', alpha=0.3)
    
    # Style the plot
    style_correlation_plot(ax, 'Feature Correlation Matrix', fig)
    
    return fig, ax