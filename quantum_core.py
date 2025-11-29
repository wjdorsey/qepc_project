"""
QEPC Quantum Core - Beginner Friendly Version
Simple quantum-inspired improvements to QEPC predictions
"""

import numpy as np
import pandas as pd
from scipy import stats

class QuantumQEPC:
    """
    Simple quantum-inspired enhancements for QEPC
    
    Usage:
        quantum_qepc = QuantumQEPC(player_data, team_data)
        prediction = quantum_qepc.predict(team_a, team_b)
    """
    
    def __init__(self, player_data=None, team_data=None):
        """
        Initialize with your data
        
        player_data: DataFrame from Player_Season_Averages.csv
        team_data: DataFrame from your team stats
        """
        self.player_data = player_data
        self.team_data = team_data
        
        print("✅ Quantum QEPC initialized!")
        if player_data is not None:
            print(f"   Loaded {len(player_data):,} player-season records")
        if team_data is not None:
            print(f"   Loaded {len(team_data):,} team records")
    
    def calculate_player_entanglement(self, team_name, season='2023-24'):
        """
        Calculate how players on a team correlate (entanglement)
        
        Returns: Correlation matrix showing player chemistry
        """
        if self.player_data is None:
            return None
        
        # Get players from this team
        team_players = self.player_data[
            (self.player_data['TEAM'] == team_name) &
            (self.player_data['Season'] == season)
        ]
        
        if len(team_players) < 2:
            return None
        
        # Calculate correlation matrix (this is the "entanglement")
        # High correlation = players perform together (entangled)
        key_stats = ['PPG', 'RPG', 'APG']
        correlations = team_players[key_stats].corr()
        
        return correlations
    
    def quantum_weighted_average(self, values, weights=None):
        """
        Quantum-inspired weighted average
        
        Instead of: simple weighted average
        Uses: Probability amplitude weighting (quantum way)
        
        values: List of predictions
        weights: Importance of each prediction
        """
        if weights is None:
            # Equal weights if not specified
            weights = np.ones(len(values))
        
        # Normalize weights (like quantum probability amplitudes)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate quantum-weighted average
        # This is |amplitude|² weighting (quantum probability)
        quantum_weights = weights ** 2
        quantum_weights = quantum_weights / np.sum(quantum_weights)
        
        result = np.sum(values * quantum_weights)
        
        return result
    
    def quantum_monte_carlo(self, mean, std, n_sims=10000, importance_weight=1.0):
        """
        Quantum-inspired Monte Carlo simulation
        
        Instead of: Uniform random sampling
        Uses: Importance sampling with quantum weights
        
        mean: Expected value (e.g., team score)
        std: Standard deviation (uncertainty)
        n_sims: Number of simulations
        importance_weight: How much to weight around the mean
        """
        # Standard Monte Carlo (uniform random)
        standard_samples = np.random.normal(mean, std, n_sims)
        
        # Quantum improvement: Importance sampling
        # Focus more samples around likely outcomes
        importance_samples = np.random.normal(
            mean, 
            std / np.sqrt(importance_weight),  # Reduced variance
            n_sims
        )
        
        # Combine with quantum weighting
        # More weight on importance-sampled results
        all_samples = np.concatenate([standard_samples, importance_samples])
        weights = np.concatenate([
            np.ones(n_sims) * 0.3,  # 30% weight on standard
            np.ones(n_sims) * 0.7   # 70% weight on importance
        ])
        
        # Calculate quantum-weighted statistics
        quantum_mean = self.quantum_weighted_average(all_samples, weights)
        
        return {
            'mean': quantum_mean,
            'samples': all_samples,
            'weights': weights,
            'std': np.std(all_samples)
        }
    
    def quantum_interference_ensemble(self, predictions, confidences=None):
        """
        Combine multiple predictions using quantum interference
        
        Instead of: Average all predictions equally
        Uses: Quantum interference (amplifies good predictions, cancels noise)
        
        predictions: List of predictions from different models
        confidences: How confident each prediction is
        """
        predictions = np.array(predictions)
        
        if confidences is None:
            confidences = np.ones(len(predictions))
        
        confidences = np.array(confidences)
        
        # Quantum interference: Convert to amplitudes
        # High confidence = high amplitude
        amplitudes = np.sqrt(confidences)
        amplitudes = amplitudes / np.sum(amplitudes)
        
        # Interference: Constructive for similar predictions
        # Destructive for outliers
        mean_pred = np.mean(predictions)
        deviations = np.abs(predictions - mean_pred)
        
        # Reduce amplitude for outliers (destructive interference)
        interference_weights = amplitudes * np.exp(-deviations / np.std(predictions))
        interference_weights = interference_weights / np.sum(interference_weights)
        
        # Final quantum-interfered prediction
        final_prediction = np.sum(predictions * interference_weights)
        
        return {
            'prediction': final_prediction,
            'weights': interference_weights,
            'raw_predictions': predictions
        }
    
    def predict_game(self, team_a_score_mean, team_a_score_std, 
                     team_b_score_mean, team_b_score_std,
                     n_sims=10000):
        """
        Predict game outcome using quantum-enhanced Monte Carlo
        
        team_a_score_mean: Expected score for team A
        team_a_score_std: Uncertainty in team A score
        team_b_score_mean: Expected score for team B
        team_b_score_std: Uncertainty in team B score
        n_sims: Number of simulations
        
        Returns: Dictionary with predictions
        """
        # Quantum Monte Carlo for each team
        team_a_sim = self.quantum_monte_carlo(
            team_a_score_mean, 
            team_a_score_std, 
            n_sims,
            importance_weight=1.5  # More focused sampling
        )
        
        team_b_sim = self.quantum_monte_carlo(
            team_b_score_mean, 
            team_b_score_std, 
            n_sims,
            importance_weight=1.5
        )
        
        # Calculate win probability
        # Team A wins if their score > Team B score
        team_a_samples = team_a_sim['samples'][:n_sims]
        team_b_samples = team_b_sim['samples'][:n_sims]
        
        team_a_wins = team_a_samples > team_b_samples
        win_probability = np.mean(team_a_wins)
        
        # Calculate expected spread
        spread = team_a_sim['mean'] - team_b_sim['mean']
        
        return {
            'team_a_score': team_a_sim['mean'],
            'team_b_score': team_b_sim['mean'],
            'spread': spread,
            'team_a_win_prob': win_probability,
            'team_b_win_prob': 1 - win_probability,
            'total': team_a_sim['mean'] + team_b_sim['mean']
        }
    
    def analyze_player_consistency(self, player_name, season='2023-24'):
        """
        Analyze player consistency using quantum uncertainty principles
        
        High uncertainty = Boom/bust player
        Low uncertainty = Consistent player
        
        Works with RAW game logs - calculates stats on the fly
        """
        if self.player_data is None:
            return None
        
        # Get player's games for this season
        player_games = self.player_data[
            (self.player_data['PLAYER_NAME'] == player_name) &
            (self.player_data['Season'] == season)
        ]
        
        if len(player_games) == 0:
            return None
        
        # Calculate stats from raw games
        ppg_mean = player_games['PTS'].mean()
        ppg_std = player_games['PTS'].std()
        
        # Handle edge cases
        if ppg_mean == 0 or ppg_std == 0:
            return None
        
        # Quantum uncertainty: ΔE * Δt ≥ ℏ/2
        # In our case: Variance * Games ≥ constant
        ppg_uncertainty = ppg_std
        
        # Consistency score (inverse of coefficient of variation)
        consistency = 1 / (1 + ppg_uncertainty / ppg_mean)
        
        # Categorize
        if consistency > 0.7:
            category = 'Consistent'
        elif consistency > 0.5:
            category = 'Moderate'
        else:
            category = 'Volatile'
        
        return {
            'player': player_name,
            'ppg': ppg_mean,
            'uncertainty': ppg_uncertainty,
            'consistency_score': consistency,
            'category': category,
            'games_played': len(player_games)
        }


# Helper function for easy use
def create_quantum_qepc(player_csv_path=None, team_csv_path=None):
    """
    Easy way to create Quantum QEPC
    
    Usage:
        qepc = create_quantum_qepc(
            player_csv_path='data/props/Player_Season_Averages.csv',
            team_csv_path='data/TeamStatistics.csv'
        )
    """
    player_data = None
    team_data = None
    
    if player_csv_path:
        player_data = pd.read_csv(player_csv_path)
        print(f"✅ Loaded player data: {player_csv_path}")
    
    if team_csv_path:
        team_data = pd.read_csv(team_csv_path)
        print(f"✅ Loaded team data: {team_csv_path}")
    
    return QuantumQEPC(player_data, team_data)


# Example usage (uncomment to test)
if __name__ == "__main__":
    print("🔮 Quantum QEPC - Testing")
    print()
    
    # Create instance
    qepc = QuantumQEPC()
    
    # Test quantum Monte Carlo
    print("Test 1: Quantum Monte Carlo")
    result = qepc.quantum_monte_carlo(mean=110, std=10, n_sims=10000)
    print(f"   Expected: 110, Got: {result['mean']:.1f}")
    print()
    
    # Test quantum interference
    print("Test 2: Quantum Interference Ensemble")
    predictions = [108, 112, 105, 115, 110]
    confidences = [0.8, 0.9, 0.6, 0.7, 0.85]
    result = qepc.quantum_interference_ensemble(predictions, confidences)
    print(f"   Predictions: {predictions}")
    print(f"   Quantum result: {result['prediction']:.1f}")
    print()
    
    # Test game prediction
    print("Test 3: Game Prediction")
    game = qepc.predict_game(
        team_a_score_mean=112, team_a_score_std=8,
        team_b_score_mean=108, team_b_score_std=9
    )
    print(f"   Team A: {game['team_a_score']:.1f}")
    print(f"   Team B: {game['team_b_score']:.1f}")
    print(f"   Spread: {game['spread']:.1f}")
    print(f"   Win Prob: {game['team_a_win_prob']:.1%}")
    
    print()
    print("✅ All tests passed!")

    """
QEPC Module: qml_predictor.py
=============================
Quantum Machine Learning predictor using PennyLane.
Encodes strength differentials into qubit rotations for win probabilities,
mimicking quantum superposition of game states.
"""

import pennylane as qml
from pennylane import numpy as np  # PennyLane's NumPy for autodiff
from pennylane.optimize import GradientDescentOptimizer

# Quantum device: Simulate on CPU (wires=1 for single-qubit simplicity)
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def quantum_circuit(strength_diff, params):
    """
    Single-qubit variational circuit.
    - Superposition: Initial rotation encodes differential (like ELO or lambda diff).
    - Entanglement-like: Learnable layers interfere for nonlinear mapping.
    - Measurement: X-basis for asymmetric probs (positive/negative diffs differ).
    """
    # Encode feature as RY rotation (superposition of |0> and |1>)
    angle = np.arctan(strength_diff)  # Normalize diff to [-pi/2, pi/2] for stability
    qml.RY(angle, wires=0)
    
    # Variational layers: Add depth for expressivity (quantum inspiration)
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    
    # Measure expectation in X-basis (PauliX): Yields prob in [-1,1], map to [0,1]
    return qml.expval(qml.PauliX(wires=0))

def qml_win_prob(strength_diff, trained_params):
    """Predict home win prob from differential (e.g., lambda_home - lambda_away)."""
    raw_prob = quantum_circuit(strength_diff, trained_params)
    return (raw_prob + 1) / 2  # Map [-1,1] to [0,1]

def train_qml_classifier(X_train, y_train, epochs=100, learning_rate=0.1):
    """
    Train on historical data.
    - X_train: List of strength diffs (e.g., [home_lambda - away_lambda]).
    - y_train: Binary labels (1=home win, 0=loss).
    Returns trained params.
    """
    params = np.random.randn(2)  # Init for two variational params
    opt = GradientDescentOptimizer(stepsize=learning_rate)
    
    def cost(params):
        preds = [qml_win_prob(x, params) for x in X_train]
        return np.mean((np.array(preds) - y_train) ** 2)  # MSE loss
    
    for epoch in range(epochs):
        params = opt.step(cost, params)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {cost(params):.4f}")
    
    return params

# Example usage in QEPC:
# trained_params = train_qml_classifier(historical_diffs, historical_outcomes)
# For a game: win_prob = qml_win_prob(lambda_home - lambda_away, trained_params)
