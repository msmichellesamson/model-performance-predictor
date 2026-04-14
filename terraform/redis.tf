# Redis cluster for ML model performance caching
resource "google_redis_instance" "model_cache" {
  name           = "model-performance-cache"
  tier           = "STANDARD_HA"
  memory_size_gb = 4
  region         = var.region
  
  redis_version     = "REDIS_6_X"
  display_name     = "Model Performance Cache"
  
  auth_enabled = true
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    notify-keyspace-events = "Ex"
  }
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
  
  labels = {
    environment = var.environment
    project     = "model-performance-predictor"
    component   = "cache"
  }
}

resource "google_redis_instance" "drift_cache" {
  name           = "drift-detection-cache"
  tier           = "BASIC"
  memory_size_gb = 2
  region         = var.region
  
  redis_version = "REDIS_6_X"
  display_name  = "Drift Detection Cache"
  
  auth_enabled = true
  
  redis_configs = {
    maxmemory-policy = "volatile-ttl"
  }
  
  labels = {
    environment = var.environment
    project     = "model-performance-predictor"
    component   = "drift-detection"
  }
}

output "redis_host" {
  value = google_redis_instance.model_cache.host
  sensitive = false
}

output "redis_port" {
  value = google_redis_instance.model_cache.port
  sensitive = false
}

output "drift_redis_host" {
  value = google_redis_instance.drift_cache.host
  sensitive = false
}