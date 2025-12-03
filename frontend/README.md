# WESAD Frontend - Docker Setup

## Quick Start

### Production Mode

```bash
# Build the production image
docker-compose build frontend

# Start the container
docker-compose up -d frontend

# Access the application
# Open http://localhost:3000 in your browser
```

### Development Mode (with hot-reload)

```bash
# Start development container
docker-compose --profile dev up -d frontend-dev

# Access the development server
# Open http://localhost:3001 in your browser
```

## Docker Commands Reference

### Production Commands

```bash
# Build production image
docker-compose build frontend

# Start production container (detached)
docker-compose up -d frontend

# Start production container (with logs)
docker-compose up frontend

# Stop container
docker-compose stop frontend

# Remove container
docker-compose down

# View logs
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f frontend

# Restart container
docker-compose restart frontend

# View container status
docker-compose ps

# Execute command in running container
docker exec -it wesad-frontend sh

# Check health status
docker inspect --format='{{.State.Health.Status}}' wesad-frontend
```

### Development Commands

```bash
# Start development container
docker-compose --profile dev up -d frontend-dev

# View development logs
docker-compose logs -f frontend-dev

# Stop development container
docker-compose --profile dev down

# Restart development container
docker-compose restart frontend-dev

# Access development container shell
docker exec -it wesad-frontend-dev sh
```

### Cleanup Commands

```bash
# Stop and remove containers
docker-compose down

# Stop and remove containers with volumes
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Complete cleanup (containers, volumes, images)
docker-compose down -v --rmi all

# Prune Docker system (be careful!)
docker system prune -a
```

### Using NPM Scripts

```bash
# Production
npm run docker:build      # Build production image
npm run docker:up         # Start production container
npm run docker:down       # Stop containers
npm run docker:logs       # View production logs

# Development
npm run docker:dev        # Start development container
npm run docker:logs-dev   # View development logs
```

## Environment Variables

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your configuration:
```bash
REACT_APP_API_URL=http://localhost:5000/api
NODE_ENV=production
```

## Port Configuration

Default ports:
- **Production**: `3000` (mapped to container port 80)
- **Development**: `3001` (mapped to container port 3000)

To change ports, edit `docker-compose.yml`:
```yaml
ports:
  - "8080:80"  # Change 3000 to 8080
```

## Troubleshooting

### Port already in use

```bash
# Check what's using the port
lsof -i :3000

# Or kill the process
kill -9 $(lsof -t -i:3000)

# Or change the port in docker-compose.yml
```

### Container won't start

```bash
# Check logs
docker-compose logs frontend

# Check container status
docker-compose ps

# Remove and rebuild
docker-compose down
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

### Hot reload not working in development

Ensure these environment variables are set in `docker-compose.yml`:
```yaml
environment:
  - CHOKIDAR_USEPOLLING=true
  - WATCHPACK_POLLING=true
```

### Build fails

```bash
# Clean build cache
docker-compose build --no-cache frontend

# Or manually clean Docker
docker system prune -a
docker-compose build frontend
```

### Can't connect to backend API

1. Check if backend is running
2. Verify `REACT_APP_API_URL` in `.env`
3. Ensure CORS is configured on backend

### Permission errors

```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# Or run with sudo (not recommended)
sudo docker-compose up -d
```

## Advanced Usage

### Build specific target

```bash
# Build development image
docker build --target development -t wesad-frontend:dev .

# Build production image
docker build --target production -t wesad-frontend:prod .
```

### Run without docker-compose

```bash
# Build
docker build -t wesad-frontend:latest .

# Run production
docker run -d \
  -p 3000:80 \
  --name wesad-frontend \
  -e REACT_APP_API_URL=http://localhost:5000/api \
  wesad-frontend:latest

# Run development
docker run -d \
  -p 3001:3000 \
  --name wesad-frontend-dev \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/public:/app/public \
  -e REACT_APP_API_URL=http://localhost:5000/api \
  -e CHOKIDAR_USEPOLLING=true \
  wesad-frontend:dev

# Stop and remove
docker stop wesad-frontend
docker rm wesad-frontend
```

### View resource usage

```bash
# Container stats
docker stats wesad-frontend

# Detailed inspect
docker inspect wesad-frontend
```

### Export/Import images

```bash
# Save image to file
docker save wesad-frontend:latest > wesad-frontend.tar

# Load image from file
docker load < wesad-frontend.tar
```

## Health Checks

The production container includes health checks:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' wesad-frontend

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' wesad-frontend
```

## Production Deployment

### Build optimized production image

```bash
docker-compose build --no-cache frontend
```

### Tag and push to registry (optional)

```bash
# Tag image
docker tag wesad-frontend:latest yourusername/wesad-frontend:1.0.0

# Push to Docker Hub
docker push yourusername/wesad-frontend:1.0.0
```

## Support

For issues and questions:
1. Check logs: `docker-compose logs -f frontend`
2. Verify configuration in `.env`
3. Ensure all dependencies are installed
4. Try rebuilding: `docker-compose build --no-cache frontend`