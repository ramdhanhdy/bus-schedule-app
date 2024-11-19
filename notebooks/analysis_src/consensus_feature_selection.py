import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor, Pool
import shap

class FeatureImportanceAnalyzer:
    """
    A comprehensive framework for ensemble feature selection using multiple methods
    and sensitivity analysis.
    """
    
    def __init__(self, style_colors=None):
        """Initialize the analyzer with visual style settings and method characteristics"""
        # Visual styling
        self.style_colors = style_colors or {
            'background': '#F7F9FC',
            'text': '#2D3B45',
            'grid': '#E1E5EA',
            'bars': ['#00B2A9', '#7C98B3', '#FF8C61', '#892B64', '#2D5D7C']
        }
        
        # Method characteristics and weights
        self.method_characteristics = {
            'catboost': {
                'type': 'tree_based',
                'handles_nonlinear': True,
                'handles_interactions': True,
                'interpretability': 'medium',
                'computational_cost': 'high'
            },
            'mutual_info': {
                'type': 'statistical',
                'handles_nonlinear': True,
                'handles_interactions': False,
                'interpretability': 'high',
                'computational_cost': 'medium'
            },
            'correlation': {
                'type': 'statistical',
                'handles_nonlinear': False,
                'handles_interactions': False,
                'interpretability': 'high',
                'computational_cost': 'low'
            },
            'permutation': {
                'type': 'model_agnostic',
                'handles_nonlinear': True,
                'handles_interactions': True,
                'interpretability': 'medium',
                'computational_cost': 'high'
            },
            'shap': {
                'type': 'model_agnostic',
                'handles_nonlinear': True,
                'handles_interactions': True,
                'interpretability': 'high',
                'computational_cost': 'high'
            }
        }
        
        # Default weight parameters
        self.default_reliability_weight = 0.6
        self.default_diversity_weight = 0.4

    def analyze_features(self, X, y, categorical_features, figsize=(15, 20), print_results=True):
        """Main method to analyze feature importance using multiple methods"""
        if print_results:
            print("Calculating feature importance using multiple methods...")
        
        # Get CatBoost importance
        if print_results:
            print("\nCalculating CatBoost importance...")
        cb_model, cb_importance = self._get_catboost_importance(X, y, categorical_features)
        
        # Get Mutual Information importance
        if print_results:
            print("Calculating Mutual Information importance...")
        mi_importance = self._get_mutual_info_importance(X, y, categorical_features)
        
        # Get Correlation importance
        if print_results:
            print("Calculating Correlation importance...")
        corr_importance = self._get_correlation_importance(X, y, categorical_features)
        
        # Get Permutation importance
        if print_results:
            print("Calculating Permutation importance...")
        perm_importance = self._get_permutation_importance(cb_model, X, y)
        
        # Get SHAP importance
        if print_results:
            print("Calculating SHAP importance...")
        shap_values, shap_importance = self._get_shap_importance(cb_model, X)
        
        results_dict = {
            'catboost': cb_importance,
            'mutual_info': mi_importance,
            'correlation': corr_importance,
            'permutation': perm_importance,
            'shap': shap_importance
        }
        
        if print_results:
            self._plot_importance_comparison(results_dict, shap_values, X, figsize)
        return results_dict

    def _normalize_importance(self, importance_scores):
        """Normalize importance scores to [0, 1] range"""
        min_val = importance_scores.min()
        max_val = importance_scores.max()
        return (importance_scores - min_val) / (max_val - min_val)

    def _get_catboost_importance(self, X, y, categorical_features):
        """Calculate CatBoost feature importance"""
        X_transformed = X.copy()
        if 'distance' in X_transformed.columns:
            X_transformed['distance'] = np.cbrt(X_transformed['distance'])
            
        cb_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            cat_features=list(categorical_features),
            verbose=False,
            random_state=42
        )
        
        train_pool = Pool(X_transformed, y, cat_features=list(categorical_features))
        cb_model.fit(train_pool)
        
        return cb_model, pd.DataFrame({
            'feature': X.columns,
            'importance': self._normalize_importance(cb_model.feature_importances_)
        }).sort_values('importance', ascending=True)

    def _get_mutual_info_importance(self, X, y, categorical_features):
        """Calculate Mutual Information importance"""
        X_transformed = X.copy()
        if 'distance' in X_transformed.columns:
            X_transformed['distance'] = np.cbrt(X_transformed['distance'])
            
        mi_scores = []
        for col in X_transformed.columns:
            if col in categorical_features:
                mi_score = mutual_info_regression(
                    pd.get_dummies(X_transformed[col]).values,
                    y,
                    random_state=42
                ).sum()
            else:
                mi_score = mutual_info_regression(
                    X_transformed[[col]].values,
                    y,
                    random_state=42
                )[0]
            mi_scores.append(mi_score)
        
        return pd.DataFrame({
            'feature': X.columns,
            'importance': self._normalize_importance(np.array(mi_scores))
        }).sort_values('importance', ascending=True)

    def _get_correlation_importance(self, X, y, categorical_features):
        """Calculate correlation-based importance"""
        X_transformed = X.copy()
        if 'distance' in X_transformed.columns:
            X_transformed['distance'] = np.cbrt(X_transformed['distance'])
            
        corr_scores = []
        for col in X_transformed.columns:
            if col in categorical_features:
                means = y.groupby(X_transformed[col]).mean()
                score = abs(pearsonr(X_transformed[col].map(means), y)[0])
            else:
                score = abs(pearsonr(X_transformed[col], y)[0])
            corr_scores.append(score)
        
        return pd.DataFrame({
            'feature': X.columns,
            'importance': self._normalize_importance(np.array(corr_scores))
        }).sort_values('importance', ascending=True)

    def _get_permutation_importance(self, model, X, y):
        """Calculate permutation importance"""
        X_transformed = X.copy()
        if 'distance' in X_transformed.columns:
            X_transformed['distance'] = np.cbrt(X_transformed['distance'])
            
        perm_importance = permutation_importance(
            model, X_transformed, y,
            n_repeats=10,
            random_state=42,
            scoring='neg_root_mean_squared_error'
        )
        
        return pd.DataFrame({
            'feature': X.columns,
            'importance': self._normalize_importance(np.abs(perm_importance.importances_mean))
        }).sort_values('importance', ascending=True)

    def _get_shap_importance(self, model, X):
        """Calculate SHAP importance"""
        X_transformed = X.copy()
        if 'distance' in X_transformed.columns:
            X_transformed['distance'] = np.cbrt(X_transformed['distance'])
            
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        
        return shap_values, pd.DataFrame({
            'feature': X.columns,
            'importance': self._normalize_importance(np.abs(shap_values).mean(axis=0))
        }).sort_values('importance', ascending=True)

    def _plot_importance_comparison(self, results_dict, shap_values, X, figsize=(15, 20)):
        """Plot comparison of feature importance from different methods"""
        # Create two separate figures
        # Figure 1: Method comparisons
        fig1 = plt.figure(figsize=(15, 15))
        gs1 = fig1.add_gridspec(3, 2)
        
        # Plot individual method results
        for idx, (method, result) in enumerate(results_dict.items()):
            ax = fig1.add_subplot(gs1[idx // 2, idx % 2])
            self._plot_importance_bars(ax, result, f"{method.capitalize()} Feature Importance", idx)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: SHAP summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X,
            plot_type="bar",
            show=True
        )

    def _plot_importance_bars(self, ax, data, title, idx):
        """Plot horizontal bar chart for feature importance"""
        ax.set_facecolor(self.style_colors['background'])
        
        # Use modulo to cycle through colors if we have more methods than colors
        color_idx = idx % len(self.style_colors['bars'])
        bars = ax.barh(data['feature'], data['importance'], 
                    color=self.style_colors['bars'][color_idx],
                    alpha=0.85)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', 
                    ha='left', va='center',
                    fontsize=8)
        
        # Customize appearance
        ax.set_title(title, pad=20, fontsize=12, color=self.style_colors['text'])
        ax.grid(True, alpha=0.3, color=self.style_colors['grid'])
        ax.tick_params(colors=self.style_colors['text'])
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.style_colors['grid'])
        ax.spines['bottom'].set_color(self.style_colors['grid'])

    def get_consensus_features(self, results, reliability_weight=0.6, diversity_weight=0.4, n_features=10):
        """Get consensus features using specified weights"""
        reliability_scores, diversity_scores = self._calculate_dynamic_weights(results)
        
        # Calculate final weights
        final_weights = {}
        for method in results.keys():
            final_weights[method] = (
                reliability_scores[method] * reliability_weight +
                diversity_scores[method] * diversity_weight
            )
        
        # Get top features for each method
        top_features = {
            method: df.sort_values('importance', ascending=False)['feature'].head(n_features).tolist()
            for method, df in results.items()
        }
        
        # Calculate weighted feature importance
        all_features = []
        for method, features in top_features.items():
            weight_multiplier = int(final_weights[method] * 20)
            all_features.extend([f for f in features for _ in range(weight_multiplier)])
        
        # Get consensus features
        feature_counts = pd.Series(all_features).value_counts()
        threshold = np.mean(feature_counts) + np.std(feature_counts)
        consensus_features = feature_counts[feature_counts >= threshold].index.tolist()
        
        self._print_analysis_results(results, top_features, consensus_features, final_weights)
        return consensus_features

    def _calculate_dynamic_weights(self, results):
        """Calculate dynamic weights based on method agreement and characteristics"""
        # Calculate correlation matrix
        importance_corr = pd.DataFrame({
            method: results[method].set_index('feature')['importance']
            for method in results.keys()
        })
        corr_matrix = importance_corr.corr(method='spearman')
        
        # Calculate diversity scores
        diversity_scores = {}
        for method in results.keys():
            other_methods = [m for m in results.keys() if m != method]
            avg_correlation = np.mean([corr_matrix.loc[method, other] for other in other_methods])
            diversity_scores[method] = 1 - abs(avg_correlation)
        
        # Calculate reliability scores
        reliability_scores = {}
        for method, chars in self.method_characteristics.items():
            score = 1.0
            if chars['handles_nonlinear']:
                score *= 1.1
            if chars['handles_interactions']:
                score *= 1.1
            if chars['interpretability'] == 'high':
                score *= 1.05
            if chars['computational_cost'] == 'low':
                score *= 1.02
            reliability_scores[method] = score
        
        # Normalize scores
        total_reliability = sum(reliability_scores.values())
        total_diversity = sum(diversity_scores.values())
        
        reliability_scores = {k: v/total_reliability for k, v in reliability_scores.items()}
        diversity_scores = {k: v/total_diversity for k, v in diversity_scores.items()}
        
        return reliability_scores, diversity_scores

    def _print_analysis_results(self, results, top_features, consensus_features, final_weights):
        """Print detailed analysis results"""
        print("\n📊 Feature Selection Analysis")
        print("=" * 50)
        print("\nMethod Weights:")
        for method, weight in final_weights.items():
            print(f"  • {method:<12}: {weight:.3f}")
        
        print("\nTop Features by Method:")
        for method, features in top_features.items():
            print(f"\n{method.capitalize()}:")
            for i, feature in enumerate(features[:5], 1):
                importance = results[method].set_index('feature').loc[feature, 'importance']
                print(f"  {i}. {feature:<20} ({importance:.3f})")
        
        print("\n🌟 Consensus Features:")
        for i, feature in enumerate(consensus_features, 1):
            methods = [m for m, f in top_features.items() if feature in f]
            print(f"  {i}. {feature:<20} (Methods: {', '.join(methods)})")

def run_feature_selection_experiment(data, target_col='duration'):
    """Run complete feature selection experiment"""
    # Prepare data
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Initialize analyzer and run analysis
    analyzer = FeatureImportanceAnalyzer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Run analysis once and store results
        results = analyzer.analyze_features(X, y, categorical_features, print_results=False)
        
        # Get consensus features with final printing
        consensus_features = analyzer.get_consensus_features(results)
        
        # Optionally run sensitivity analysis
        sensitivity_results = analyzer.perform_sensitivity_analysis(results)
    
    return consensus_features, results, sensitivity_results

# Usage example
if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='Converting `np.inexact` or `np.floating` to a dtype is deprecated')
    
    # Run feature selection
    consensus_features, results, sensitivity_results = run_feature_selection_experiment(filtered_df)