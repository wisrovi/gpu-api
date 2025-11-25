# Kubernetes Manifests

This directory contains all the Kubernetes manifests required to deploy the System Metrics API to a Kubernetes cluster.

## Prerequisites

-   A running Kubernetes cluster.
-   The NVIDIA Device Plugin for Kubernetes must be installed on the cluster to expose GPU resources to pods. The `nvidia-device-plugin.yaml` manifest is provided for this purpose.
-   A container image of the application pushed to a registry accessible by your cluster.

## Files

-   `deployment.yaml`: Defines the deployment for the API server, including resource requests/limits (including `nvidia.com/gpu`), probes, and affinity rules.
-   `service.yaml`: Defines a `ClusterIP` service to expose the deployment within the cluster.
-   `nvidia-device-plugin.yaml`: The manifest to deploy the official NVIDIA device plugin daemonset.
-   `device-plugin-configmap.yaml`: ConfigMap for the NVIDIA device plugin, if you need to customize its behavior.

## Deployment

To deploy the application, you can use `kubectl`:

```bash
# Make sure your image name is correct in deployment.yaml
kubectl apply -f .
```

Refer to the main `README.md` in the root of the project for more detailed instructions.
