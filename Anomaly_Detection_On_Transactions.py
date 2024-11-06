import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import datetime
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

class TransactionAnomalyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.lof = LocalOutlierFactor(contamination=0.1, novelty=True)
        self.autoencoder = None
        
    def prepare_features(self, df):
        """
        Prepare transaction data features for anomaly detection
        """
        # Convert timestamp to datetime features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate transaction frequency features
        df['transaction_count'] = df.groupby('account_id')['amount'].transform('count')
        df['daily_transaction_count'] = df.groupby(['account_id', df['timestamp'].dt.date])['amount'].transform('count')
        
        # Calculate amount-based features
        df['daily_total'] = df.groupby(['account_id', df['timestamp'].dt.date])['amount'].transform('sum')
        df['amount_mean'] = df.groupby('account_id')['amount'].transform('mean')
        df['amount_std'] = df.groupby('account_id')['amount'].transform('std')
        df['amount_zscore'] = (df['amount'] - df['amount_mean']) / df['amount_std']
        
        # Handle NaN values
        df = df.fillna(0)
        
        return df
    
    def build_autoencoder(self, input_dim):
        """
        Build autoencoder model for anomaly detection
        """
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Create model
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        return self.autoencoder
    
    def detect_anomalies(self, df, methods=['isolation_forest', 'lof', 'autoencoder', 'statistical']):
        """
        Detect anomalies using multiple methods
        """
        # Prepare features for modeling
        feature_columns = ['amount', 'hour', 'day_of_week', 'is_weekend', 
                         'transaction_count', 'daily_transaction_count', 
                         'daily_total', 'amount_zscore']
        
        X = df[feature_columns].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        results = pd.DataFrame()
        
        # Isolation Forest
        if 'isolation_forest' in methods:
            results['isolation_forest'] = self.isolation_forest.fit_predict(X_scaled)
        
        # Local Outlier Factor
        if 'lof' in methods:
            self.lof.fit(X_scaled)
            results['lof'] = self.lof.predict(X_scaled)
        
        # Autoencoder
        if 'autoencoder' in methods:
            if self.autoencoder is None:
                self.build_autoencoder(len(feature_columns))
            
            # Train autoencoder
            self.autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)
            
            # Predict and calculate reconstruction error
            predictions = self.autoencoder.predict(X_scaled)
            reconstruction_error = np.mean(np.square(X_scaled - predictions), axis=1)
            results['autoencoder'] = (reconstruction_error > np.percentile(reconstruction_error, 90)).astype(int)
        
        # Statistical method (Z-score based)
        if 'statistical' in methods:
            results['statistical'] = (abs(df['amount_zscore']) > 3).astype(int)
        
        # Combine results
        results['combined_score'] = results.mean(axis=1)
        results['is_anomaly'] = (results['combined_score'] > 0.5).astype(int)
        
        return results
    
    def analyze_anomalies(self, df, results):
        """
        Analyze detected anomalies and generate insights
        """
        df_anomalies = df[results['is_anomaly'] == 1].copy()
        
        analysis = {
            'total_transactions': len(df),
            'total_anomalies': len(df_anomalies),
            'anomaly_rate': len(df_anomalies) / len(df) * 100,
            'average_anomaly_amount': df_anomalies['amount'].mean(),
            'max_anomaly_amount': df_anomalies['amount'].max(),
            'common_anomaly_hours': df_anomalies['hour'].value_counts().head(),
            'common_anomaly_days': df_anomalies['day_of_week'].value_counts().head(),
            'affected_accounts': df_anomalies['account_id'].nunique()
        }
        
        # Pattern analysis
        analysis['weekend_anomaly_rate'] = (
            df_anomalies['is_weekend'].mean() / df['is_weekend'].mean()
        )
        
        # Risk scoring
        df_anomalies['risk_score'] = (
            df_anomalies['amount_zscore'].abs() * 
            results.loc[df_anomalies.index, 'combined_score']
        )
        
        analysis['high_risk_transactions'] = len(df_anomalies[
            df_anomalies['risk_score'] > df_anomalies['risk_score'].quantile(0.9)
        ])
        
        return analysis

def generate_sample_data(n_samples=1000):
    """
    Generate sample transaction data for testing
    """
    np.random.seed(42)
    
    # Normal transactions
    data = {
        'account_id': np.random.randint(1, 101, n_samples),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
        'amount': np.random.normal(100, 30, n_samples),
        'merchant_category': np.random.randint(1, 21, n_samples)
    }
    
    # Add some anomalies
    n_anomalies = int(n_samples * 0.05)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Large amount anomalies
    data['amount'][anomaly_indices[:n_anomalies//2]] *= 10
    
    # Unusual time anomalies
    unusual_hours = np.random.randint(1, 5, n_anomalies//2)
    data['timestamp'][anomaly_indices[n_anomalies//2:]] = pd.date_range(
        start='2024-01-01 02:00:00', 
        periods=n_anomalies//2, 
        freq='H'
    )
    
    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    transactions_df = generate_sample_data()
    
    # Initialize detector
    detector = TransactionAnomalyDetector()
    
    # Prepare features
    transactions_df = detector.prepare_features(transactions_df)
    
    # Detect anomalies
    results = detector.detect_anomalies(transactions_df)
    
    # Analyze results
    analysis = detector.analyze_anomalies(transactions_df, results)
    
    # Print summary
    print("Anomaly Detection Results:")
    print(f"Total Transactions: {analysis['total_transactions']}")
    print(f"Total Anomalies: {analysis['total_anomalies']}")
    print(f"Anomaly Rate: {analysis['anomaly_rate']:.2f}%")
    print(f"Average Anomaly Amount: ${analysis['average_anomaly_amount']:.2f}")
    print(f"High Risk Transactions: {analysis['high_risk_transactions']}")
