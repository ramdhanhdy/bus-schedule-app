import matplotlib.pyplot as plt
import pandas as pd
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
                  '═══ Strong: |r| > 0.5\n'
                  '---- Moderate: 0.3 < |r| < 0.5\n'
                  '.... Weak: |r| < 0.3')
    
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

def visualize_model_comparison(model_results):
    """Enhanced visualization with clearer metric interpretation"""
    if not model_results:
        print("\nNo model results available for comparison")
        return
    
    # Style configuration
    plt.style.use('bmh')
    colors = {
        'primary': ['#2E86AB', '#A23B72'],
        'grid': '#E1E5EA',
        'text': '#2D3B45',
        'accent': '#FF8C61',
        'positive': '#2ecc71',  # Green for positive changes
        'negative': '#e74c3c'   # Red for negative changes
    }
    
    # Prepare metrics data with better organization and direction
    metrics = {
        'R-squared': {
            'train': 'train_r2', 
            'test': 'test_r2',
            'format': '.4f',
            'higher_is_better': True
        },
        'RMSE': {
            'train': 'train_rmse', 
            'test': 'test_rmse',
            'format': '.2f',
            'higher_is_better': False
        },
        'MAE': {
            'train': 'train_mae', 
            'test': 'test_mae',
            'format': '.2f',
            'higher_is_better': False
        }
    }
    
    # Create figure with better layout
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
    
    # 1. Enhanced Bar plot with separate subplots for different metric types
    ax1 = fig.add_subplot(gs[0, :])
    comparison_data = []

    # Prepare data
    for model_type, results in model_results.items():
        if results:
            for metric_name, metric_info in metrics.items():
                for split in ['train', 'test']:
                    key = metric_info[split]
                    comparison_data.append({
                        'Model': results['feature_set'],
                        'Metric': f"{metric_name} ({split.title()})",
                        'Value': results[key],
                        'Higher_is_better': metric_info['higher_is_better']
                    })

    df_plot = pd.DataFrame(comparison_data)

    # Create the bar plot
    bars = sns.barplot(
        data=df_plot,
        x='Metric',
        y='Value',
        hue='Model',
        ax=ax1,
        palette=colors['primary'],
        alpha=0.8
    )

    # Add percentage differences above bars with color coding
    unique_metrics = df_plot['Metric'].unique()
    for i, metric in enumerate(unique_metrics):
        metric_data = df_plot[df_plot['Metric'] == metric]
        if len(metric_data) == 2:
            base_value = metric_data[metric_data['Model'] == 'Without Weather Features']['Value'].iloc[0]
            weather_value = metric_data[metric_data['Model'] == 'With Weather Features']['Value'].iloc[0]
            higher_is_better = metric_data['Higher_is_better'].iloc[0]
            
            # Calculate percentage difference
            pct_diff = ((weather_value - base_value) / base_value) * 100
            
            # Determine if this change is good or bad
            is_improvement = (pct_diff > 0) == higher_is_better
            
            # Get the higher bar's height for positioning
            max_height = max(base_value, weather_value)
            
            # Add percentage text with color coding
            ax1.text(
                i,  # x position
                max_height * 1.05,  # y position
                f'{"+" if pct_diff > 0 else "−"} {abs(pct_diff):.1f} %',
                ha='center',
                va='bottom',
                color=colors['positive'] if is_improvement else colors['negative'],
                fontsize=10,
                fontweight='bold'
            )
            
            # Add small arrow or indicator of desired direction
            direction_marker = '▲' if higher_is_better else '▼'
            ax1.text(
                i,  # x position
                ax1.get_ylim()[1] * 0.02,  # y position at bottom
                direction_marker,
                ha='center',
                va='bottom',
                color=colors['text'],
                alpha=0.5,
                fontsize=8
            )

    # Rest of the plot styling
    # Update the title section in the visualization
    ax1.set_title('Comparative Analysis: Weather vs. Non-Weather Feature Models for Journey Time Prediction\n(↑ Optimal for R², ↓ Optimal for RMSE/MAE)', 
              pad=20, 
              fontsize=14, 
              color=colors['text'],
              fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    ax1.set_ylabel('Score', fontsize=12)

    # Adjust y-axis limit
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.15)
        
    # 2. Enhanced Error Distribution Plot
    ax2 = fig.add_subplot(gs[1, 0])
    for i, (model_type, results) in enumerate(model_results.items()):
        if results:
            errors = results['predictions']['y_test'] - results['predictions']['y_pred_test']
            sns.kdeplot(
                data=errors,
                label=results['feature_set'],
                ax=ax2,
                color=colors['primary'][i],
                fill=True,
                alpha=0.3
            )
    ax2.axvline(x=0, color=colors['text'], linestyle='--', alpha=0.5)
    ax2.set_title('Prediction Error Distribution', fontsize=12, color=colors['text'])
    ax2.set_xlabel('Error (minutes)', fontsize=10)
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    
    # 3. Enhanced Scatter plot
    ax3 = fig.add_subplot(gs[1, 1])
    for i, (model_type, results) in enumerate(model_results.items()):
        if results:
            sns.scatterplot(
                x=results['predictions']['y_test'],
                y=results['predictions']['y_pred_test'],
                label=results['feature_set'],
                alpha=0.5,
                ax=ax3,
                color=colors['primary'][i]
            )
    
    # Improved perfect prediction line
    lims = [
        min(ax3.get_xlim()[0], ax3.get_ylim()[0]),
        max(ax3.get_xlim()[1], ax3.get_ylim()[1])
    ]
    ax3.plot(lims, lims, '--', color=colors['text'], alpha=0.75, label='Perfect Prediction')
    ax3.set_title('Actual vs Predicted Duration', fontsize=12, color=colors['text'])
    ax3.set_xlabel('Actual Duration (minutes)', fontsize=10)
    ax3.set_ylabel('Predicted Duration (minutes)', fontsize=10)
    ax3.grid(True, alpha=0.3, color=colors['grid'])
    
    # 4. Add Residual Plot
    ax4 = fig.add_subplot(gs[2, 0])
    for i, (model_type, results) in enumerate(model_results.items()):
        if results:
            sns.regplot(
                x=results['predictions']['y_test'],
                y=results['predictions']['y_test'] - results['predictions']['y_pred_test'],
                label=results['feature_set'],
                ax=ax4,
                color=colors['primary'][i],
                scatter_kws={'alpha':0.5},
                line_kws={'color': colors['primary'][i]}
            )
    ax4.axhline(y=0, color=colors['text'], linestyle='--', alpha=0.5)
    ax4.set_title('Residual Analysis', fontsize=12, color=colors['text'])
    ax4.set_xlabel('Actual Duration (minutes)', fontsize=10)
    ax4.set_ylabel('Residual (minutes)', fontsize=10)
    ax4.grid(True, alpha=0.3, color=colors['grid'])
    
    # 5. Add Error Quantile Plot
    ax5 = fig.add_subplot(gs[2, 1])
    for i, (model_type, results) in enumerate(model_results.items()):
        if results:
            errors = np.abs(results['predictions']['y_test'] - results['predictions']['y_pred_test'])
            quantiles = np.percentile(errors, np.arange(0, 101, 1))
            ax5.plot(np.arange(0, 101, 1), quantiles, 
                    label=results['feature_set'],
                    color=colors['primary'][i])
    ax5.set_title('Error Quantile Distribution', fontsize=12, color=colors['text'])
    ax5.set_xlabel('Percentile', fontsize=10)
    ax5.set_ylabel('Absolute Error (minutes)', fontsize=10)
    ax5.grid(True, alpha=0.3, color=colors['grid'])

    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Enhanced metrics table with styling
    print("\n📊 Detailed Model Comparison:")
    print("=" * 100)
    
    comparison_metrics = pd.DataFrame([
        {
            'Model': results['feature_set'],
            'Train R²': f"{results['train_r2']:.4f}",
            'Test R²': f"{results['test_r2']:.4f}",
            'Train RMSE': f"{results['train_rmse']:.2f}",
            'Test RMSE': f"{results['test_rmse']:.2f}",
            'Train MAE': f"{results['train_mae']:.2f}",
            'Test MAE': f"{results['test_mae']:.2f}",
            'Parameters': len(results['best_params'])
        }
        for results in model_results.values() if results
    ])
    
    print(comparison_metrics.to_string(index=False))
    
    # Enhanced statistical analysis
    print("\n📈 Detailed Error Analysis:")
    print("=" * 100)
    for model_type, results in model_results.items():
        if results:
            errors = results['predictions']['y_test'] - results['predictions']['y_pred_test']
            abs_errors = np.abs(errors)
            
            print(f"\n🔍 {results['feature_set']}:")
            print(f"  • Mean Error: {errors.mean():.2f} ± {errors.std():.2f} minutes")
            print(f"  • Median Absolute Error: {np.median(abs_errors):.2f} minutes")
            print(f"  • Error Range: [{errors.min():.2f}, {errors.max():.2f}] minutes")
            print(f"  • Error Quartiles: {np.percentile(errors, [25, 50, 75])}")
            print(f"  • 90th percentile of absolute error: {np.percentile(abs_errors, 90):.2f} minutes")
            print(f"  • 95th percentile of absolute error: {np.percentile(abs_errors, 95):.2f} minutes")
            print(f"  • Skewness: {pd.Series(errors).skew():.3f}")
            print(f"  • Kurtosis: {pd.Series(errors).kurtosis():.3f}")