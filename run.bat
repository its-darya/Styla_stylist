@echo off
echo Starting Styla Stylist...

:: Check if Docker is running, if not start Docker Desktop
docker info >nul 2>&1
if not errorlevel 1 goto DOCKER_RUNNING

echo Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo Waiting for Docker to be ready (this may take a minute)...

:WAIT_DOCKER
timeout /t 3 /nobreak >nul
docker info >nul 2>&1
if errorlevel 1 goto WAIT_DOCKER
echo Docker is ready!
goto DOCKER_READY

:DOCKER_RUNNING
echo Docker is already running.

:DOCKER_READY
echo Starting Database containers...
docker compose up -d

echo Starting Backend...
start "Styla Backend" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo Starting Frontend...
start "Styla Frontend" cmd /k "cd frontend && npm run dev"

echo Both servers and database are starting!
