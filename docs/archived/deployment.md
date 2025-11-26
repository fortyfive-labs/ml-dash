# Deployment Guide

This guide covers deploying your own ML-Dash server for remote mode operation.

## Overview

ML-Dash's remote mode requires a backend server that provides:
- **REST API** for experiment, log, parameter, metric, and file operations
- **MongoDB** for metadata and structured data storage
- **S3-compatible storage** (AWS S3, MinIO, etc.) for file storage
- **JWT authentication** for secure access

```{note}
Local mode requires no deployment - just install the SDK and start metricing!
```

## Prerequisites

### System Requirements

- **OS**: Linux, macOS, or Windows with WSL
- **Memory**: Minimum 2GB RAM, recommended 4GB+
- **Disk**: 10GB+ for application and data
- **Network**: Inbound access on port 3000 (or your chosen port)

### Required Services

1. **Node.js**: v18+ (for server runtime)
2. **MongoDB**: v5.0+ (for data storage)
3. **S3 Storage**: AWS S3, MinIO, or compatible service

## Quick Start with Docker Compose

The fastest way to deploy ML-Dash server is using Docker Compose.

### 1. Create docker-compose.yml

```yaml
version: '3.8'

services:
  # ML-Dash API Server
  ml-dash-server:
    image: ml-dash/server:latest  # Replace with actual image
    container_name: ml-dash-server
    ports:
      - "3000:3000"
    environment:
      # Server Configuration
      PORT: 3000
      NODE_ENV: production

      # MongoDB Connection
      MONGODB_URI: mongodb://mongo:27017/ml-dash

      # S3 Configuration (using MinIO)
      S3_ENDPOINT: http://minio:9000
      S3_ACCESS_KEY: minioadmin
      S3_SECRET_KEY: minioadmin
      S3_BUCKET: ml-dash-files
      S3_REGION: us-east-1

      # JWT Configuration
      JWT_SECRET: your-secret-key-change-this-in-production
      JWT_EXPIRATION: 30d

    depends_on:
      - mongo
      - minio
    restart: unless-stopped
    networks:
      - ml-dash-network

  # MongoDB Database
  mongo:
    image: mongo:6.0
    container_name: ml-dash-mongo
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db
    environment:
      MONGO_INITDB_DATABASE: ml-dash
    restart: unless-stopped
    networks:
      - ml-dash-network

  # MinIO S3-compatible Storage
  minio:
    image: minio/minio:latest
    container_name: ml-dash-minio
    ports:
      - "9000:9000"      # API
      - "9001:9001"      # Console
    volumes:
      - minio-data:/data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    restart: unless-stopped
    networks:
      - ml-dash-network

  # MinIO Client - Create bucket on startup
  minio-setup:
    image: minio/mc:latest
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      until /usr/bin/mc alias set myminio http://minio:9000 minioadmin minioadmin; do
        echo 'Waiting for MinIO...'
        sleep 1
      done;
      /usr/bin/mc mb myminio/ml-dash-files || true;
      /usr/bin/mc anonymous set download myminio/ml-dash-files;
      exit 0;
      "
    networks:
      - ml-dash-network

volumes:
  mongo-data:
  minio-data:

networks:
  ml-dash-network:
    driver: bridge
```

### 2. Configure Environment

Create a `.env` file for sensitive configuration:

```bash
# .env
JWT_SECRET=your-super-secret-jwt-key-change-this
MONGODB_URI=mongodb://mongo:27017/ml-dash
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ml-dash-server

# Stop services
docker-compose down
```

### 4. Verify Deployment

```bash
# Check health endpoint
curl http://localhost:3000/health

# Expected response:
# {"status":"ok"}
```

### 5. Test Connection

```python
from ml_dash import Experiment

with Experiment(
    name="test-experiment",
    project="test-project",
    remote="http://localhost:3000",
    user_name="test-user"
) as experiment:
    experiment.log("Deployment successful!")
    print(f"Experiment ID: {experiment.id}")
```

## Manual Deployment

If you prefer to deploy without Docker, follow these steps.

### 1. Install Node.js

```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS
brew install node@18

# Verify installation
node --version  # Should be v18+
npm --version
```

### 2. Setup MongoDB

**Option A: Docker**
```bash
docker run -d \
  --name ml-dash-mongo \
  -p 27017:27017 \
  -v mongo-data:/data/db \
  mongo:6.0
```

**Option B: Native Installation**
```bash
# Ubuntu/Debian
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
```

### 3. Setup MinIO (or AWS S3)

**Option A: MinIO (Self-hosted)**
```bash
# Download MinIO
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio

# Start MinIO
./minio server /mnt/data --console-address ":9001"

# Create bucket
mc alias set myminio http://localhost:9000 minioadmin minioadmin
mc mb myminio/ml-dash-files
```

**Option B: AWS S3**
```bash
# Create S3 bucket
aws s3 mb s3://ml-dash-files --region us-east-1

# Configure IAM user with S3 access
# Use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
```

### 4. Install ML-Dash Server

```bash
# Clone repository (replace with actual repo)
git clone https://github.com/your-org/ml-dash-server.git
cd ml-dash-server

# Install dependencies
npm install

# Build (if TypeScript)
npm run build
```

### 5. Configure Environment Variables

Create `.env` file:

```bash
# Server Configuration
PORT=3000
NODE_ENV=production
LOG_LEVEL=info

# MongoDB
MONGODB_URI=mongodb://localhost:27017/ml-dash

# S3 Storage
S3_ENDPOINT=http://localhost:9000  # MinIO endpoint (omit for AWS)
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
S3_BUCKET=ml-dash-files
S3_REGION=us-east-1
S3_FORCE_PATH_STYLE=true  # Required for MinIO

# JWT Authentication
JWT_SECRET=your-secret-key-change-this-in-production
JWT_EXPIRATION=30d

# Optional: CORS
CORS_ORIGIN=*  # Or specific domain: https://yourdomain.com
```

### 6. Start Server

```bash
# Development mode
npm run dev

# Production mode
npm run start

# Or with PM2 for process management
npm install -g pm2
pm2 start dist/server.js --name ml-dash-server
pm2 save
pm2 startup
```

## Cloud Deployment

### AWS Deployment

#### Using ECS (Fargate)

1. **Build and push Docker image**:
```bash
# Build image
docker build -t ml-dash-server .

# Tag for ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag ml-dash-server:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-dash-server:latest

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-dash-server:latest
```

2. **Create ECS Task Definition** (JSON):
```json
{
  "family": "ml-dash-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "ml-dash-server",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/ml-dash-server:latest",
      "portMappings": [{"containerPort": 3000}],
      "environment": [
        {"name": "PORT", "value": "3000"},
        {"name": "MONGODB_URI", "value": "mongodb://..."},
        {"name": "S3_BUCKET", "value": "ml-dash-files"},
        {"name": "S3_REGION", "value": "us-east-1"}
      ],
      "secrets": [
        {"name": "JWT_SECRET", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "S3_ACCESS_KEY", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "S3_SECRET_KEY", "valueFrom": "arn:aws:secretsmanager:..."}
      ]
    }
  ]
}
```

3. **Use DocumentDB** for MongoDB:
```bash
# Create DocumentDB cluster
aws docdb create-db-cluster \
  --db-cluster-identifier ml-dash-cluster \
  --engine docdb \
  --master-username admin \
  --master-user-password <password>
```

#### Using EC2

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# 3. Install dependencies and deploy (same as Manual Deployment)

# 4. Configure security group
# - Allow inbound TCP 3000 from your IP/CIDR
# - Allow inbound TCP 22 for SSH
```

### Google Cloud Platform (GCP)

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-dash-server

# Deploy to Cloud Run
gcloud run deploy ml-dash-server \
  --image gcr.io/PROJECT_ID/ml-dash-server \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "MONGODB_URI=...,S3_BUCKET=..." \
  --set-secrets "JWT_SECRET=ml-dash-jwt-secret:latest"
```

### Kubernetes Deployment

Create `kubernetes/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-dash-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-dash-server
  template:
    metadata:
      labels:
        app: ml-dash-server
    spec:
      containers:
      - name: ml-dash-server
        image: ml-dash/server:latest
        ports:
        - containerPort: 3000
        env:
        - name: PORT
          value: "3000"
        - name: MONGODB_URI
          valueFrom:
            secretKeyRef:
              name: ml-dash-secrets
              key: mongodb-uri
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: ml-dash-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-dash-service
spec:
  selector:
    app: ml-dash-server
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/secrets.yaml  # Create secrets first
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | 3000 | Server port |
| `NODE_ENV` | No | development | Environment (development/production) |
| `LOG_LEVEL` | No | info | Logging level (debug/info/warn/error) |
| `MONGODB_URI` | Yes | - | MongoDB connection string |
| `S3_ENDPOINT` | No | - | S3 endpoint (for MinIO/custom) |
| `S3_ACCESS_KEY` | Yes | - | S3 access key |
| `S3_SECRET_KEY` | Yes | - | S3 secret key |
| `S3_BUCKET` | Yes | - | S3 bucket name |
| `S3_REGION` | No | us-east-1 | S3 region |
| `S3_FORCE_PATH_STYLE` | No | false | Use path-style URLs (for MinIO) |
| `JWT_SECRET` | Yes | - | JWT signing secret |
| `JWT_EXPIRATION` | No | 30d | JWT token expiration |
| `CORS_ORIGIN` | No | * | CORS allowed origins |
| `MAX_FILE_SIZE` | No | 100MB | Maximum file upload size |

## Authentication Configuration

### JWT Secret

The `JWT_SECRET` is critical for security. Generate a strong secret:

```bash
# Generate strong random secret
openssl rand -base64 64

# Or use Node.js
node -e "console.log(require('crypto').randomBytes(64).toString('base64'))"
```

**Important**: This secret must match between:
1. The server's `JWT_SECRET` environment variable
2. The SDK's token generation (when using `user_name`)

### Development vs Production

**Development** (user_name auto-generation):
```python
Experiment(
    remote="http://localhost:3000",
    user_name="alice"  # Generates JWT automatically
)
```

**Production** (proper authentication):
```python
# User authenticates via your auth service
# Your service returns JWT token
api_key = authenticate_user("alice", "password")

Experiment(
    remote="https://ml-dash.yourcompany.com",
    api_key=api_key  # Use real JWT token
)
```

## Health Checks & Monitoring

### Health Endpoint

```bash
# Check server health
curl http://localhost:3000/health

# Response when healthy:
{
  "status": "ok",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "uptime": 3600,
  "mongodb": "connected",
  "s3": "connected"
}
```

### Logging

Configure logging level:
```bash
# Environment variable
LOG_LEVEL=debug  # debug, info, warn, error

# View logs (Docker)
docker-compose logs -f ml-dash-server

# View logs (PM2)
pm2 logs ml-dash-server
```

### Monitoring Metrics

Monitor these key metrics:
- **Request rate**: Requests per second
- **Response time**: Average and P95 latency
- **Error rate**: 4xx and 5xx responses
- **Database connections**: MongoDB connection pool
- **Storage usage**: S3 bucket size
- **Memory/CPU**: Server resource usage

## Backup & Recovery

### MongoDB Backup

```bash
# Backup with mongodump
mongodump --uri="mongodb://localhost:27017/ml-dash" --out=/backup/$(date +%Y%m%d)

# Restore
mongorestore --uri="mongodb://localhost:27017/ml-dash" /backup/20240101
```

### S3 Backup

```bash
# Sync to backup bucket (AWS)
aws s3 sync s3://ml-dash-files s3://ml-dash-files-backup

# Sync from MinIO
mc mirror myminio/ml-dash-files s3-backup/ml-dash-files
```

### Automated Backups

Add to crontab:
```bash
# Daily MongoDB backup at 2 AM
0 2 * * * /usr/local/bin/backup-ml-dash.sh

# Weekly S3 sync
0 3 * * 0 /usr/local/bin/sync-s3-backup.sh
```

## Troubleshooting

### Server Won't Start

**Check logs**:
```bash
docker-compose logs ml-dash-server
# Or
pm2 logs ml-dash-server
```

**Common issues**:
1. MongoDB not accessible - check `MONGODB_URI`
2. S3 credentials invalid - verify `S3_ACCESS_KEY` and `S3_SECRET_KEY`
3. Port already in use - change `PORT` environment variable

### Cannot Connect from SDK

1. **Check server is running**:
   ```bash
   curl http://localhost:3000/health
   ```

2. **Verify network access**:
   - Local: Check firewall rules
   - Cloud: Check security groups/firewall rules

3. **Check CORS settings**:
   - Set `CORS_ORIGIN` to allow your client domain

### MongoDB Connection Issues

```bash
# Test MongoDB connection
mongosh "mongodb://localhost:27017/ml-dash"

# Check MongoDB logs
docker logs ml-dash-mongo
```

### S3 Connection Issues

```bash
# Test S3 connection (AWS CLI)
aws s3 ls s3://ml-dash-files --region us-east-1

# Test MinIO connection
mc ls myminio/ml-dash-files
```

## Security Best Practices

1. **Use HTTPS** in production:
   - Setup reverse proxy (nginx, Caddy)
   - Use SSL/TLS certificates (Let's Encrypt)

2. **Secure MongoDB**:
   - Enable authentication
   - Use network encryption
   - Restrict network access

3. **Secure S3**:
   - Use IAM roles (AWS)
   - Enable bucket encryption
   - Implement access policies

4. **Environment Variables**:
   - Never commit `.env` to git
   - Use secrets management (AWS Secrets Manager, Vault)
   - Rotate secrets regularly

5. **Network Security**:
   - Use VPC/private networks
   - Implement rate limiting
   - Setup DDoS protection

## Scaling

### Horizontal Scaling

ML-Dash server is stateless and can be scaled horizontally:

```bash
# Docker Compose
docker-compose up --scale ml-dash-server=3

# Kubernetes
kubectl scale deployment ml-dash-server --replicas=5
```

### Load Balancing

Use a load balancer to distribute traffic:

**Nginx**:
```nginx
upstream ml-dash {
    server localhost:3001;
    server localhost:3002;
    server localhost:3003;
}

server {
    listen 80;
    location / {
        proxy_pass http://ml-dash;
    }
}
```

**AWS Application Load Balancer** handles this automatically.

## Next Steps

- ✅ Server deployed and running
- → [Getting Started](getting-started.md) - Start using ML-Dash
- → [Architecture](architecture.md) - Understand the internals
- → [FAQ](faq.md) - Common questions and issues

## Support

Having deployment issues? Check:
- [FAQ & Troubleshooting](faq.md)
- [GitHub Issues](https://github.com/your-org/ml-dash/issues)
- Community Discord/Slack
