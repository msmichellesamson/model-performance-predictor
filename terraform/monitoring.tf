variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "prometheus_retention_days" {
  description = "Prometheus data retention in days"
  type        = number
  default     = 15
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
}

resource "google_compute_instance" "prometheus" {
  name         = "${var.environment}-model-perf-prometheus"
  machine_type = "e2-standard-2"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }

  metadata_startup_script = templatefile("${path.module}/scripts/prometheus_setup.sh", {
    retention_days = var.prometheus_retention_days
    environment    = var.environment
  })

  service_account {
    email  = google_service_account.monitoring_sa.email
    scopes = ["cloud-platform"]
  }

  tags = ["prometheus", "monitoring", var.environment]

  labels = {
    environment = var.environment
    service     = "prometheus"
    project     = "model-performance-predictor"
  }
}

resource "google_compute_instance" "grafana" {
  name         = "${var.environment}-model-perf-grafana"
  machine_type = "e2-standard-2"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }

  metadata_startup_script = templatefile("${path.module}/scripts/grafana_setup.sh", {
    admin_password    = var.grafana_admin_password
    prometheus_url    = "http://${google_compute_instance.prometheus.network_interface[0].network_ip}:9090"
    environment       = var.environment
  })

  service_account {
    email  = google_service_account.monitoring_sa.email
    scopes = ["cloud-platform"]
  }

  tags = ["grafana", "monitoring", var.environment]

  labels = {
    environment = var.environment
    service     = "grafana"
    project     = "model-performance-predictor"
  }
}

resource "google_service_account" "monitoring_sa" {
  account_id   = "${var.environment}-monitoring-sa"
  display_name = "Model Performance Monitoring Service Account"
}

resource "google_project_iam_member" "monitoring_metrics_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.monitoring_sa.email}"
}

resource "google_project_iam_member" "monitoring_viewer" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.monitoring_sa.email}"
}

resource "google_compute_firewall" "prometheus" {
  name    = "${var.environment}-prometheus-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["9090"]
  }

  source_ranges = ["10.0.0.0/8"]
  target_tags   = ["prometheus"]
}

resource "google_compute_firewall" "grafana" {
  name    = "${var.environment}-grafana-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["3000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["grafana"]
}

resource "google_compute_firewall" "node_exporter" {
  name    = "${var.environment}-node-exporter-firewall"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["9100"]
  }

  source_ranges = ["10.0.0.0/8"]
  target_tags   = ["monitoring"]
}

resource "google_compute_disk" "prometheus_data" {
  name = "${var.environment}-prometheus-data"
  zone = "${var.region}-a"
  size = 100
  type = "pd-ssd"

  labels = {
    environment = var.environment
    service     = "prometheus"
  }
}

resource "google_compute_attached_disk" "prometheus_data" {
  disk     = google_compute_disk.prometheus_data.id
  instance = google_compute_instance.prometheus.id
}

resource "google_compute_disk" "grafana_data" {
  name = "${var.environment}-grafana-data"
  zone = "${var.region}-a"
  size = 20
  type = "pd-standard"

  labels = {
    environment = var.environment
    service     = "grafana"
  }
}

resource "google_compute_attached_disk" "grafana_data" {
  disk     = google_compute_disk.grafana_data.id
  instance = google_compute_instance.grafana.id
}

resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "${var.environment} Model Predictor High CPU"
  combiner     = "OR"

  conditions {
    display_name = "CPU usage above 80%"

    condition_threshold {
      filter         = "resource.type=\"compute_instance\" AND resource.labels.instance_name=~\"${var.environment}-model-perf-.*\""
      duration       = "300s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = 0.8

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "memory_usage" {
  display_name = "${var.environment} Model Predictor High Memory"
  combiner     = "OR"

  conditions {
    display_name = "Memory usage above 85%"

    condition_threshold {
      filter         = "resource.type=\"compute_instance\" AND resource.labels.instance_name=~\"${var.environment}-model-perf-.*\""
      duration       = "300s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = 0.85

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]
}

resource "google_monitoring_alert_policy" "disk_usage" {
  display_name = "${var.environment} Model Predictor High Disk Usage"
  combiner     = "OR"

  conditions {
    display_name = "Disk usage above 90%"

    condition_threshold {
      filter         = "resource.type=\"compute_instance\" AND resource.labels.instance_name=~\"${var.environment}-model-perf-.*\""
      duration       = "600s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = 0.9

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "${var.environment} Model Performance Alerts"
  type         = "email"

  labels = {
    email_address = "ops@company.com"
  }
}

resource "google_storage_bucket" "monitoring_config" {
  name     = "${var.project_id}-${var.environment}-monitoring-config"
  location = var.region

  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    environment = var.environment
    service     = "monitoring"
  }
}

resource "google_storage_bucket_object" "prometheus_config" {
  name   = "prometheus.yml"
  bucket = google_storage_bucket.monitoring_config.name
  content = templatefile("${path.module}/configs/prometheus.yml", {
    environment = var.environment
    targets = [
      "${google_compute_instance.prometheus.network_interface[0].network_ip}:9090",
      "${google_compute_instance.grafana.network_interface[0].network_ip}:3000"
    ]
  })
}

resource "google_storage_bucket_object" "grafana_datasources" {
  name   = "grafana-datasources.yml"
  bucket = google_storage_bucket.monitoring_config.name
  content = templatefile("${path.module}/configs/grafana-datasources.yml", {
    prometheus_url = "http://${google_compute_instance.prometheus.network_interface[0].network_ip}:9090"
  })
}

resource "google_storage_bucket_object" "grafana_dashboards" {
  name   = "model-performance-dashboard.json"
  bucket = google_storage_bucket.monitoring_config.name
  source = "${path.module}/configs/model-performance-dashboard.json"
}

output "prometheus_ip" {
  value = google_compute_instance.prometheus.network_interface[0].access_config[0].nat_ip
}

output "grafana_ip" {
  value = google_compute_instance.grafana.network_interface[0].access_config[0].nat_ip
}

output "grafana_url" {
  value = "http://${google_compute_instance.grafana.network_interface[0].access_config[0].nat_ip}:3000"
}

output "prometheus_url" {
  value = "http://${google_compute_instance.prometheus.network_interface[0].access_config[0].nat_ip}:9090"
}

output "monitoring_sa_email" {
  value = google_service_account.monitoring_sa.email
}