#!/usr/bin/env python3
"""
Synthetic inference data generator for ML model performance testing.

This script generates realistic synthetic inference data with various patterns:
- Normal operations with gradual drift
- Sudden performance degradation
- Data quality issues
- Seasonal patterns
- Anomalous behavior

Used for testing the model performance predictor system.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import sys

import numpy as np
import structlog
from scipy import stats
import httpx
import asyncpg


logger = structlog.get_logger()


class DataGenerationError(Exception):
    """Custom exception for data generation errors."""
    pass


@dataclass
class ModelMetrics:
    """Model inference metrics."""
    model_id: str
    timestamp: datetime
    prediction_latency_ms: float
    confidence_score: float
    input_features: Dict[str, Union[float, int, str]]
    prediction: Union[str, float, int]
    actual: Optional[Union[str, float, int]] = None
    drift_score: Optional[float] = None
    data_quality_score: Optional[float] = None


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""
    model_ids: List[str]
    duration_hours: int
    samples_per_hour: int
    drift_probability: float
    anomaly_probability: float
    degradation_probability: float
    feature_names: List[str]
    prediction_type: str  # 'classification' or 'regression'
    output_format: str  # 'api', 'database', 'file'
    api_endpoint: Optional[str] = None
    db_connection_string: Optional[str] = None
    output_file: Optional[str] = None


class SyntheticDataGenerator:
    """Generates realistic synthetic ML inference data."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.start_time = datetime.utcnow()
        self.baseline_latency = 50.0  # ms
        self.baseline_confidence = 0.85
        self.drift_accumulator = 0.0
        self.performance_degradation = 1.0
        
    def generate_feature_vector(self, drift_level: float = 0.0) -> Dict[str, Union[float, int]]:
        """Generate synthetic feature vector with optional drift."""
        features = {}
        
        for i, feature_name in enumerate(self.config.feature_names):
            if feature_name.endswith('_categorical'):
                # Categorical feature with drift affecting distribution
                categories = ['A', 'B', 'C', 'D']
                if drift_level > 0.3:
                    # Shift distribution during drift
                    weights = [0.1, 0.2, 0.3, 0.4]
                else:
                    weights = [0.25, 0.25, 0.25, 0.25]
                features[feature_name] = np.random.choice(categories, p=weights)
            
            elif feature_name.endswith('_numeric'):
                # Numeric feature with normal distribution
                base_mean = i * 10
                base_std = 2.0
                
                # Add drift
                drift_shift = drift_level * 5.0
                drift_scale = 1.0 + drift_level * 0.5
                
                value = np.random.normal(
                    base_mean + drift_shift,
                    base_std * drift_scale
                )
                features[feature_name] = round(value, 3)
            
            else:
                # Default numeric feature
                features[feature_name] = round(np.random.uniform(0, 100), 2)
        
        return features

    def calculate_prediction_latency(self, 
                                   base_features: Dict,
                                   is_anomaly: bool = False) -> float:
        """Calculate realistic prediction latency based on various factors."""
        base_latency = self.baseline_latency * self.performance_degradation
        
        # Feature complexity affects latency
        feature_complexity = len([k for k in base_features.keys() if isinstance(base_features[k], str)])
        complexity_multiplier = 1.0 + (feature_complexity * 0.1)
        
        # Random variation
        noise = np.random.lognormal(0, 0.3)
        
        # Anomalies can cause higher latency
        anomaly_multiplier = random.uniform(2.0, 5.0) if is_anomaly else 1.0
        
        latency = base_latency * complexity_multiplier * noise * anomaly_multiplier
        
        # Add occasional spikes (simulating GC, network issues, etc.)
        if random.random() < 0.05:
            latency *= random.uniform(3.0, 10.0)
        
        return max(1.0, latency)  # Minimum 1ms

    def generate_prediction(self, features: Dict) -> Tuple[Union[str, float], float]:
        """Generate prediction and confidence score."""
        if self.config.prediction_type == 'classification':
            # Simulate binary classification
            classes = ['positive', 'negative']
            
            # Simple feature-based logic for consistency
            feature_sum = sum(v for v in features.values() if isinstance(v, (int, float)))
            probability = 1 / (1 + np.exp(-(feature_sum - 50) / 10))  # Sigmoid
            
            prediction = 'positive' if probability > 0.5 else 'negative'
            confidence = max(probability, 1 - probability)
            
            # Add noise and drift effects
            confidence *= (1.0 - self.drift_accumulator * 0.2)
            confidence *= self.performance_degradation
            confidence = max(0.1, min(0.99, confidence))
            
        else:  # regression
            # Generate continuous prediction
            feature_sum = sum(v for v in features.values() if isinstance(v, (int, float)))
            prediction = feature_sum * 2 + np.random.normal(0, 5)
            
            # Confidence as inverse of prediction uncertainty
            uncertainty = abs(np.random.normal(0, 1)) * (1 + self.drift_accumulator)
            confidence = 1 / (1 + uncertainty)
            confidence *= self.performance_degradation
            confidence = max(0.1, min(0.99, confidence))
        
        return prediction, confidence

    def generate_ground_truth(self, 
                            prediction: Union[str, float], 
                            features: Dict,
                            delay_probability: float = 0.8) -> Optional[Union[str, float]]:
        """Generate ground truth with realistic delay patterns."""
        if random.random() > delay_probability:
            return None  # No ground truth yet
        
        if self.config.prediction_type == 'classification':
            # Ground truth should correlate with prediction but with some error
            if prediction == 'positive':
                actual = 'positive' if random.random() > 0.15 else 'negative'
            else:
                actual = 'negative' if random.random() > 0.15 else 'positive'
            
            # Performance degradation affects accuracy
            if self.performance_degradation < 0.8 and random.random() < 0.3:
                actual = 'positive' if actual == 'negative' else 'negative'
                
            return actual
        else:
            # Regression ground truth with noise
            noise_std = 10.0 * (2.0 - self.performance_degradation)
            return prediction + np.random.normal(0, noise_std)

    def calculate_drift_score(self, features: Dict) -> float:
        """Calculate drift score based on feature distribution."""
        # Simplified drift calculation
        feature_values = [v for v in features.values() if isinstance(v, (int, float))]
        if not feature_values:
            return 0.0
        
        # Compare to expected baseline (mean=50 for synthetic data)
        mean_diff = abs(np.mean(feature_values) - 50) / 50
        std_ratio = np.std(feature_values) / 20  # Expected std=20
        
        drift_score = (mean_diff + abs(std_ratio - 1.0)) / 2
        drift_score += self.drift_accumulator
        
        return min(1.0, drift_score)

    def calculate_data_quality_score(self, features: Dict) -> float:
        """Calculate data quality score."""
        quality_score = 1.0
        
        # Check for missing values (represented as None or NaN)
        missing_ratio = sum(1 for v in features.values() if v is None) / len(features)
        quality_score -= missing_ratio * 0.5
        
        # Check for outliers in numeric features
        numeric_values = [v for v in features.values() if isinstance(v, (int, float))]
        if numeric_values:
            z_scores = np.abs(stats.zscore(numeric_values))
            outlier_ratio = sum(1 for z in z_scores if z > 3) / len(z_scores)
            quality_score -= outlier_ratio * 0.3
        
        # Add random quality issues
        if random.random() < 0.1:  # 10% chance of quality issues
            quality_score *= random.uniform(0.5, 0.9)
        
        return max(0.0, min(1.0, quality_score))

    def update_system_state(self):
        """Update drift accumulation and performance degradation."""
        # Gradual drift accumulation
        if random.random() < self.config.drift_probability:
            self.drift_accumulator += random.uniform(0.01, 0.05)
            self.drift_accumulator = min(1.0, self.drift_accumulator)
        
        # Sudden performance degradation
        if random.random() < self.config.degradation_probability:
            self.performance_degradation *= random.uniform(0.8, 0.95)
            logger.info("Performance degradation simulated", 
                       degradation_level=self.performance_degradation)
        
        # Occasional recovery
        if random.random() < 0.05:
            self.performance_degradation = min(1.0, self.performance_degradation * 1.1)
            self.drift_accumulator *= 0.9

    def generate_sample(self, model_id: str, timestamp: datetime) -> ModelMetrics:
        """Generate a single inference sample."""
        # Determine if this sample has anomalies
        is_anomaly = random.random() < self.config.anomaly_probability
        
        # Generate features with current drift level
        features = self.generate_feature_vector(self.drift_accumulator)
        
        # Calculate prediction and metrics
        prediction, confidence = self.generate_prediction(features)
        latency = self.calculate_prediction_latency(features, is_anomaly)
        drift_score = self.calculate_drift_score(features)
        quality_score = self.calculate_data_quality_score(features)
        
        # Generate ground truth (may be None)
        actual = self.generate_ground_truth(prediction, features)
        
        return ModelMetrics(
            model_id=model_id,
            timestamp=timestamp,
            prediction_latency_ms=round(latency, 2),
            confidence_score=round(confidence, 4),
            input_features=features,
            prediction=prediction,
            actual=actual,
            drift_score=round(drift_score, 4),
            data_quality_score=round(quality_score, 4)
        )

    async def generate_batch(self, batch_size: int) -> List[ModelMetrics]:
        """Generate a batch of samples."""
        samples = []
        current_time = datetime.utcnow()
        
        for i in range(batch_size):
            model_id = random.choice(self.config.model_ids)
            # Spread samples across time
            sample_time = current_time + timedelta(seconds=i)
            
            sample = self.generate_sample(model_id, sample_time)
            samples.append(sample)
        
        # Update system state after each batch
        self.update_system_state()
        
        return samples


class DataOutputHandler:
    """Handles output of generated data to various destinations."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.http_client = None
        self.db_pool = None
        
    async def initialize(self):
        """Initialize output handlers."""
        if self.config.output_format == 'api' and self.config.api_endpoint:
            self.http_client = httpx.AsyncClient()
            
        elif self.config.output_format == 'database' and self.config.db_connection_string:
            self.db_pool = await asyncpg.create_pool(
                self.config.db_connection_string,
                min_size=1,
                max_size=5
            )
    
    async def send_to_api(self, samples: List[ModelMetrics]):
        """Send samples to API endpoint."""
        if not self.http_client:
            raise DataGenerationError("HTTP client not initialized")
        
        for sample in samples:
            payload = {
                'model_id': sample.model_id,
                'timestamp': sample.timestamp.isoformat(),
                'prediction_latency_ms': sample.prediction_latency_ms,
                'confidence_score': sample.confidence_score,
                'input_features': sample.input_features,
                'prediction': sample.prediction,
                'actual': sample.actual,
                'drift_score': sample.drift_score,
                'data_quality_score': sample.data_quality_score
            }
            
            try:
                response = await self.http_client.post(
                    f"{self.config.api_endpoint}/inference",
                    json=payload,
                    timeout=5.0
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error("Failed to send sample to API", error=str(e))
    
    async def send_to_database(self, samples: List[ModelMetrics]):
        """Send samples to database."""
        if not self.db_pool:
            raise DataGenerationError("Database pool not initialized")
        
        query = """
        INSERT INTO inference_logs (
            model_id, timestamp, prediction_latency_ms, confidence_score,
            input_features, prediction, actual, drift_score, data_quality_score
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        async with self.db_pool.acquire() as conn:
            for sample in samples:
                try:
                    await conn.execute(
                        query,
                        sample.model_id,
                        sample.timestamp,
                        sample.prediction_latency_ms,
                        sample.confidence_score,
                        json.dumps(sample.input_features),
                        json.dumps(sample.prediction),
                        json.dumps(sample.actual) if sample.actual else None,
                        sample.drift_score,
                        sample.data_quality_score
                    )
                except asyncpg.PostgresError as e:
                    logger.error("Failed to insert sample", error=str(e))
    
    def save_to_file(self, samples: List[ModelMetrics]):
        """Save samples to file."""
        if not self.config.output_file:
            raise DataGenerationError("Output file not specified")
        
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'a') as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample), default=str) + '\n')
    
    async def send_batch(self, samples: List[ModelMetrics]):
        """Send batch to configured output."""
        if self.config.output_format == 'api':
            await self.send_to_api(samples)
        elif self.config.output_format == 'database':
            await self.send_to_database(samples)
        elif self.config.output_format == 'file':
            self.save_to_file(samples)
        else:
            raise DataGenerationError(f"Unknown output format: {self.config.output_format}")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()
        if self.db_pool:
            await self.db_pool.close()


async def generate_data_stream(config: GenerationConfig):
    """Main data generation loop."""
    generator = SyntheticDataGenerator(config)
    output_handler = DataOutputHandler(config)
    
    try:
        await output_handler.initialize()
        
        total_samples = config.duration_hours * config.samples_per_hour
        batch_size = min(100, config.samples_per_hour // 4)  # Process in batches
        samples_per_second = config.samples_per_hour / 3600
        
        logger.info("Starting data generation",
                   total_samples=total_samples,
                   batch_size=batch_size,
                   duration_hours=config.duration_hours)
        
        generated_count = 0
        start_time = time.time()
        
        while generated_count < total_samples:
            batch_start = time.time()
            
            # Generate batch
            current_batch_size = min(batch_size, total_samples - generated_count)
            samples = await generator.generate_batch(current_batch_size)
            
            # Send to output
            await output_handler.send_batch(samples)
            
            generated_count += len(samples)
            
            # Rate limiting
            batch_duration = time.time() - batch_start
            expected_duration = current_batch_size / samples_per_second
            
            if batch_duration < expected_duration:
                await asyncio.sleep(expected_duration - batch_duration)
            
            # Progress logging
            if generated_count % (batch_size * 10) == 0:
                elapsed = time.time() - start_time
                rate = generated_count / elapsed if elapsed > 0 else 0
                logger.info("Generation progress",
                           generated=generated_count,
                           total=total_samples,
                           rate_per_second=round(rate, 2),
                           drift_level=round(generator.drift_accumulator, 3),
                           performance_level=round(generator.performance_degradation, 3))
        
        logger.info("Data generation completed",
                   total_generated=generated_count,
                   duration_seconds=round(time.time() - start_time, 2))
        
    finally:
        await output_handler.cleanup()


def create_config_from_args(args) -> GenerationConfig:
    """Create configuration from command line arguments."""
    return GenerationConfig(
        model_ids=args.model_ids,
        duration_hours=args.duration_hours,
        samples_per_hour=args.samples_per_hour,
        drift_probability=args.drift_probability,
        anomaly_probability=args.anomaly_probability,
        degradation_probability=args.degradation_probability,
        feature_names=args.feature_names,
        prediction_type=args.prediction_type,
        output_format=args.output_format,
        api_endpoint=args.api_endpoint,
        db_connection_string=args.db_connection_string,
        output_file=args.output_file
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic ML inference data")
    
    parser.add_argument('--model-ids', nargs='+', 
                       default=['fraud_detection_v1', 'recommendation_engine_v2'],
                       help='Model IDs to generate data for')
    parser.add_argument('--duration-hours', type=int, default=1,
                       help='Duration of data generation in hours')
    parser.add_argument('--samples-per-hour', type=int, default=3600,
                       help='Number of samples to generate per hour')
    parser.add_argument('--drift-probability', type=float, default=0.01,
                       help='Probability of drift in each batch (0.0-1.0)')
    parser.add_argument('--anomaly-probability', type=float, default=0.05,
                       help='Probability of anomalous samples (0.0-1.0)')
    parser.add_argument('--degradation-probability', type=float, default=0.001,
                       help='Probability of performance degradation (0.0-1.0)')
    parser.add_argument('--feature-names', nargs='+',
                       default=['user_age_numeric', 'transaction_amount_numeric', 
                               'merchant_category_categorical', 'time_of_day_numeric'],
                       help='Feature names to generate')
    parser.add_argument('--prediction-type', choices=['classification', 'regression'],
                       default='classification', help='Type of ML problem')
    parser.add_argument('--output-format', choices=['api', 'database', 'file'],
                       default='file', help='Output destination')
    parser.add_argument('--api-endpoint', help='API endpoint URL for output')
    parser.add_argument('--db-connection-string', 
                       help='PostgreSQL connection string')
    parser.add_argument('--output-file', default='synthetic_data.jsonl',
                       help='Output file path')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Validation
    if args.output_format == 'api' and not args.api_endpoint:
        logger.error("API endpoint required for API output format")
        sys.exit(1)
    
    if args.output_format == 'database' and not args.db_connection_string:
        logger.error("Database connection string required for database output format")
        sys.exit(1)
    
    try:
        config = create_config_from_args(args)
        asyncio.run(generate_data_stream(config))
    except KeyboardInterrupt:
        logger.info("Data generation interrupted by user")
    except Exception as e:
        logger.error("Data generation failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()