# System Metrics API

This project provides a FastAPI-based web service to monitor NVIDIA GPU and CPU metrics.

## Project Structure

The project is organized into the following directories:

-   `app/`: Contains the FastAPI application source code, its `Dockerfile`, and Python dependencies. See the [app/README.md](./app/README.md) for more details.
-   `k8s_files/`: Contains all the Kubernetes manifests needed to deploy the application. See the [k8s_files/README.md](./k8s_files/README.md) for more details.
-   `docker-compose.yaml`: A Docker Compose file to easily build and run the application locally with GPU support.
-   `setup-nvkind-cluster.sh`: A script to set up a local KinD cluster with NVIDIA GPU support.

## Running the Application

To run the application locally using Docker, use the following command:

```bash
docker-compose up --build
```

The API will be accessible at `http://localhost:8000`. You can find the interactive documentation at `http://localhost:8000/docs`.

## API Endpoints

-   `GET /`: Returns a simple welcome message.
-   `GET /health`: Health check endpoint for Kubernetes probes.
-   `GET /machine`: Returns a combined object with all CPU and GPU metrics.
-   `GET /cpu`: Returns detailed metrics for the system's CPU.
-   `GET /gpus`: Returns a list of all available GPUs and their metrics.
-   `GET /gpus/{gpu_id}`: Returns detailed metrics for a specific GPU by its ID.

## Deployment

For deployment to Kubernetes, refer to the instructions in the [k8s_files/README.md](./k8s_files/README.md).
