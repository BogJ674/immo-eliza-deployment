#!/bin/bash

# Start the FastAPI application
echo "Starting Immo Eliza API..."
cd api && uvicorn app:app --reload --host 0.0.0.0 --port 8000
