# -*- coding: utf-8 -*-
"""
FastAPI application to provide GPU information and metrics.
"""

import os
import subprocess
from typing import List

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --- Configuration ---
# Load configuration from environment variables with sane defaults.
HOST = os.getenv("GPU_API_HOST", "0.0.0.0")
PORT = int(os.getenv("GPU_API_PORT", "8000"))
NVIDIA_SMI_PATH = os.getenv("NVIDIA_SMI_PATH", "nvidia-smi")

# --- Pydantic Models ---

class CPUInfo(BaseModel):
    """Detailed information and metrics for the CPU."""
    usage_percent: float = Field(
        ..., description="System-wide CPU utilization as a percentage.", example=50.5
    )
    physical_cores: int = Field(
        ..., description="Number of physical CPU cores.", example=8
    )
    logical_cores: int = Field(
        ..., description="Number of logical CPU cores (including hyper-threading).", example=16
    )
    frequency_current: float = Field(
        ..., description="Current CPU frequency in Mhz.", example=3400.0
    )
    frequency_min: float = Field(
        ..., description="Minimum CPU frequency in Mhz.", example=800.0
    )
    frequency_max: float = Field(
        ..., description="Maximum CPU frequency in Mhz.", example=4800.0
    )


class GPUInfo(BaseModel):
    """Detailed information and metrics for a single GPU."""
    gpu_id: int = Field(..., description="The index of the GPU.", example=0)
    product_name: str = Field(
        ..., description="The official product name of the GPU.", example="NVIDIA GeForce RTX 3080"
    )
    memory_total: int = Field(
        ..., description="Total installed GPU memory (in MiB).", example=10240
    )
    memory_free: int = Field(
        ..., description="Unallocated GPU memory (in MiB).", example=8192
    )
    memory_used: int = Field(
        ..., description="Allocated GPU memory (in MiB).", example=2048
    )
    temperature: int = Field(
        ..., description="Current GPU temperature (in degrees C).", example=65
    )
    utilization_gpu: int = Field(
        ..., description="Percent of time over the past second the GPU was busy.", example=75
    )
    utilization_memory: int = Field(
        ..., description="Percent of time over the past second memory IO was busy.", example=30
    )

class MachineInfo(BaseModel):
    """Combined information for the machine's CPU and GPUs."""
    cpu: CPUInfo
    gpus: List[GPUInfo]


# --- FastAPI Application ---

app = FastAPI(
    title="System Metrics API",
    description="A professional API to monitor NVIDIA GPU and CPU metrics.",
    version="1.2.0",
)


# --- Helper Functions ---

def _get_cpu_info() -> CPUInfo:
    """
    Retrieves detailed metrics for the system's CPU.
    
    Raises:
        HTTPException: If there's an error fetching CPU info.
    """
    try:
        usage = psutil.cpu_percent(interval=0.1)
        freq = psutil.cpu_freq()
        
        # Handle cases where freq can be None or not have all attributes
        current_freq = freq.current if freq else 0.0
        min_freq = freq.min if freq else 0.0
        max_freq = freq.max if freq else 0.0

        return CPUInfo(
            usage_percent=usage,
            physical_cores=psutil.cpu_count(logical=False),
            logical_cores=psutil.cpu_count(logical=True),
            frequency_current=current_freq,
            frequency_min=min_freq,
            frequency_max=max_freq,
        )
    except Exception as e:
        # Re-raise as HTTPException to be handled by FastAPI
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching CPU info: {e}")

def _query_nvidia_smi() -> List[GPUInfo]:
    """
    Executes nvidia-smi to query GPU details and returns a list of GPUInfo objects.
    
    Raises:
        HTTPException: If nvidia-smi command is not found or fails.
    """
    try:
        command = [
            NVIDIA_SMI_PATH,
            "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"'{NVIDIA_SMI_PATH}' command not found. Please ensure NVIDIA drivers and nvidia-smi are installed and in your system's PATH.",
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing nvidia-smi: {e.stderr.strip()}",
        )

    output = result.stdout.strip()
    if not output:
        return []

    gpu_list = []
    for line in output.splitlines():
        try:
            (
                gpu_id,
                name,
                mem_total,
                mem_free,
                mem_used,
                temp,
                util_gpu,
                util_mem,
            ) = line.split(", ")
            
            gpu_list.append(
                GPUInfo(
                    gpu_id=int(gpu_id),
                    product_name=name,
                    memory_total=int(mem_total),
                    memory_free=int(mem_free),
                    memory_used=int(mem_used),
                    temperature=int(temp),
                    utilization_gpu=int(util_gpu),
                    utilization_memory=int(util_mem),
                )
            )
        except (ValueError, IndexError) as e:
            # Skip corrupted lines from nvidia-smi output
            print(f"Warning: Could not parse nvidia-smi output line: '{line}'. Error: {e}")
            continue
            
    return gpu_list


# --- API Endpoints ---

@app.get("/", summary="Root endpoint", tags=["General"])
def read_root():
    """Returns a welcome message."""
    return {"message": "Welcome to the System Metrics API"}


@app.get("/health", summary="Health Check", tags=["General"])
def health_check():
    """
    Simple health check endpoint that returns a 200 OK response.
    Used for Kubernetes liveness and readiness probes.
    """
    return {"status": "ok"}


@app.get("/machine", response_model=MachineInfo, summary="Get All Machine Information", tags=["Machine"])
def get_machine_info():
    """
    Retrieves a combination of all CPU and GPU metrics for the entire machine.
    """
    # Note: Error handling is done within the helper functions
    cpu_info = _get_cpu_info()
    gpu_info = _query_nvidia_smi()
    
    return MachineInfo(cpu=cpu_info, gpus=gpu_info)


@app.get("/cpu", response_model=CPUInfo, summary="Get CPU Information", tags=["CPU"])
def get_cpu_info_endpoint():
    """
    Retrieves detailed metrics for the system's CPU.
    """
    return _get_cpu_info()


@app.get(
    "/gpus",
    response_model=List[GPUInfo],
    summary="Get All GPU Information",
    tags=["GPU"],
)
def get_all_gpus():
    """
    Retrieves a list of all available NVIDIA GPUs and their current metrics.
    
    Returns an empty list if no GPUs are found.
    """
    # The original endpoint's logic can now just call the helper
    return _query_nvidia_smi()


@app.get(
    "/gpus/{gpu_id}",
    response_model=GPUInfo,
    summary="Get Specific GPU Information",
    tags=["GPU"],
)
def get_gpu_by_id(gpu_id: int):
    """
    Retrieves detailed metrics for a specific GPU by its ID (index).
    
    Raises a 404 error if the GPU with the specified ID is not found.
    """
    try:
        gpus = _query_nvidia_smi()
        for gpu in gpus:
            if gpu.gpu_id == gpu_id:
                return gpu
        raise HTTPException(status_code=404, detail=f"GPU with id {gpu_id} not found.")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# --- Main Entry Point ---

if __name__ == "__main__":
    print(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)

