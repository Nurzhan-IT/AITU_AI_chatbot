#!/bin/bash
# Run on the VPS after uploading the project: bash deploy.sh

set -e

echo "=== AITU Chatbot — VPS Deploy Script ==="

# 1. Install Docker if missing
if ! command -v docker &> /dev/null; then
    echo "[1/5] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "Docker installed. Re-login or run: newgrp docker"
fi

# 2. Add swap if less than 1GB swap exists
SWAP_TOTAL=$(free -m | awk '/^Swap:/{print $2}')
if [ "$SWAP_TOTAL" -lt 1024 ]; then
    echo "[2/5] Creating 2GB swap file..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
else
    echo "[2/5] Swap already configured (${SWAP_TOTAL}MB), skipping."
fi

# 3. Create required directories
echo "[3/5] Creating data directories..."
mkdir -p pdfs data qdrant_data

# 4. Check .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Upload your .env file to the project directory first."
    exit 1
fi

# 5. Build and start
echo "[4/5] Building and starting services..."
docker compose -f docker-compose.prod.yml pull qdrant nginx
docker compose -f docker-compose.prod.yml build bot
docker compose -f docker-compose.prod.yml up -d

echo "[5/5] Waiting for services to start..."
sleep 10

docker compose -f docker-compose.prod.yml ps
echo ""
echo "=== Deploy complete! ==="
echo "Bot logs: docker compose -f docker-compose.prod.yml logs -f bot"
echo "All logs: docker compose -f docker-compose.prod.yml logs -f"
