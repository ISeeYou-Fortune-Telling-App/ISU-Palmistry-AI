# Palmistry AI Service - Makefile
# Commands: build, start, down, log, restart, help

# Variables
COMPOSE_FILE = docker/docker-compose.yaml
SERVICE_NAME = palmistry-api

# Default target
.DEFAULT_GOAL := help

# Build the Docker image
build:
	@echo "Building Palmistry API Docker image..."
	@docker-compose -f $(COMPOSE_FILE) build
	@echo "✅ Build completed successfully!"

# Start the service
start:
	@echo "Starting Palmistry API service..."
	@docker-compose -f $(COMPOSE_FILE) up -d
	@echo "🚀 Service started at http://localhost:8080"
	@echo "📚 API docs available at http://localhost:8080/docs"

# Stop and remove containers
down:
	@echo "Stopping Palmistry API service..."
	@docker-compose -f $(COMPOSE_FILE) down
	@echo "🛑 Service stopped successfully!"

# Show service logs
log:
	@echo "Showing Palmistry API logs..."
	@docker-compose -f $(COMPOSE_FILE) logs -f $(SERVICE_NAME)

# Restart the service
restart: down start
	@echo "🔄 Service restarted successfully!"

# Show help information
help:
	@echo "Palmistry AI Service - Available Commands:"
	@echo ""
	@echo "  build     Build the Docker image"
	@echo "  start     Start the service (available at http://localhost:8080)"
	@echo "  down      Stop and remove containers"
	@echo "  log       Show service logs (follow mode)"
	@echo "  restart   Restart the service (down + start)"
	@echo "  help      Show this help message"
	@echo ""
	@echo "📚 API Documentation: http://localhost:8080/docs"
	@echo "❤️  Health Check: http://localhost:8080/health"

.PHONY: build start down log restart help
