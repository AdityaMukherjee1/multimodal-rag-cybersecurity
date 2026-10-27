"""Dynamic System and Hardware Telemetry for Cybersecurity Multimodal RAG.

Extracts real, unadulterated hardware telemetry directly from the operating system,
NVIDIA drivers, PyTorch CUDA runtime, and system processes.

NO MOCK DATA:
If a metric cannot be obtained from the OS or hardware, it returns 'Unavailable'
or 'Not supported by this hardware'.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class CPUTelemetry:
    """CPU telemetry data."""
    model_name: str
    architecture: str
    physical_cores: int | str
    logical_cores: int | str
    utilization_pct: float
    frequency_mhz: float | str
    temperature_celsius: str = "Unavailable (Not exposed by OS kernel)"


@dataclass
class RAMTelemetry:
    """System RAM telemetry data."""
    total_gb: float
    used_gb: float
    available_gb: float
    utilization_pct: float


@dataclass
class GPUTelemetry:
    """NVIDIA GPU telemetry data."""
    detected: bool
    name: str = "Not detected"
    device_index: int | str = "N/A"
    cuda_available: bool = False
    pytorch_cuda_enabled: bool = False
    compute_device: str = "CPU"
    vram_total_mb: float | str = "N/A"
    vram_used_mb: float | str = "N/A"
    vram_free_mb: float | str = "N/A"
    utilization_pct: float | str = "Unavailable"
    temperature_celsius: float | str = "Unavailable"
    power_wattage: float | str = "Unavailable"
    driver_version: str = "N/A"
    cuda_runtime_version: str = "N/A"


@dataclass
class DiskTelemetry:
    """Disk storage telemetry for the project filesystem."""
    mount_point: str
    total_gb: float
    used_gb: float
    free_gb: float
    utilization_pct: float
    read_mb: float | str = "Unavailable"
    write_mb: float | str = "Unavailable"
    temperature_celsius: str = "Unavailable (Not exposed by OS without storage controller API)"


@dataclass
class ProcessTelemetry:
    """Project-specific process resource consumption."""
    pid: int
    process_name: str
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    threads_count: int
    io_read_mb: float | str = "Unavailable"
    io_write_mb: float | str = "Unavailable"
    gpu_vram_allocated_mb: float | str = "N/A"
    gpu_vram_reserved_mb: float | str = "N/A"


class SystemTelemetryService:
    """Real-time hardware inspection and telemetry extraction service."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self._process = psutil.Process(os.getpid())
        # Prime CPU percent measurements
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    def get_cpu_telemetry(self) -> CPUTelemetry:
        """Extract CPU specifications and utilization."""
        model_name = "Unknown CPU"

        # On Windows, retrieve human-readable processor brand from Registry
        if sys.platform == "win32":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                )
                brand, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                model_name = brand.strip()
            except Exception:
                model_name = platform.processor() or "Unknown CPU"
        else:
            model_name = platform.processor() or "Unknown CPU"

        phys = psutil.cpu_count(logical=False) or "Unavailable"
        logical = psutil.cpu_count(logical=True) or "Unavailable"
        util = psutil.cpu_percent(interval=None)

        freq_info = psutil.cpu_freq()
        freq = round(freq_info.current, 1) if freq_info and freq_info.current else "Unavailable"

        return CPUTelemetry(
            model_name=model_name,
            architecture=platform.machine(),
            physical_cores=phys,
            logical_cores=logical,
            utilization_pct=util,
            frequency_mhz=freq,
            temperature_celsius="Unavailable (Not exposed by OS kernel)",
        )

    def get_ram_telemetry(self) -> RAMTelemetry:
        """Extract system memory utilization."""
        mem = psutil.virtual_memory()
        return RAMTelemetry(
            total_gb=round(mem.total / (1024**3), 2),
            used_gb=round(mem.used / (1024**3), 2),
            available_gb=round(mem.available / (1024**3), 2),
            utilization_pct=round(mem.percent, 1),
        )

    def get_gpu_telemetry(self) -> GPUTelemetry:
        """Extract dynamic GPU information and telemetry via PyTorch and nvidia-smi."""
        cuda_avail = torch.cuda.is_available()
        pytorch_cuda = bool(torch.version.cuda is not None)

        if not cuda_avail:
            return GPUTelemetry(
                detected=False,
                name="Not detected",
                cuda_available=False,
                pytorch_cuda_enabled=pytorch_cuda,
                compute_device="CPU",
                vram_total_mb="N/A",
                vram_used_mb="N/A",
                vram_free_mb="N/A",
                utilization_pct="N/A",
                temperature_celsius="N/A",
                power_wattage="N/A",
                driver_version="N/A",
                cuda_runtime_version="N/A",
            )

        # PyTorch detects at least one CUDA device
        device_name = torch.cuda.get_device_name(0)
        total_vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)

        # Query nvidia-smi for real-time hardware telemetry (utilization, temp, power)
        smi_util = "Unavailable"
        smi_temp = "Unavailable"
        smi_power = "Unavailable"
        smi_used_mb = round(torch.cuda.memory_allocated(0) / (1024**2), 1)
        smi_free_mb = round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**2), 1)
        driver_ver = "N/A"
        cuda_ver = torch.version.cuda or "N/A"

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=driver_version,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                if len(parts) >= 6:
                    driver_ver = parts[0]
                    try:
                        smi_util = float(parts[1])
                    except ValueError:
                        smi_util = parts[1]
                    try:
                        smi_used_mb = float(parts[2])
                    except ValueError:
                        pass
                    try:
                        smi_free_mb = float(parts[3])
                    except ValueError:
                        pass
                    try:
                        smi_temp = float(parts[4])
                    except ValueError:
                        smi_temp = parts[4]
                    try:
                        smi_power = float(parts[5])
                    except ValueError:
                        smi_power = parts[5]
        except Exception as e:
            logger.debug("nvidia-smi query failed or timed out: %s", e)

        return GPUTelemetry(
            detected=True,
            name=device_name,
            device_index=0,
            cuda_available=True,
            pytorch_cuda_enabled=True,
            compute_device="cuda:0",
            vram_total_mb=total_vram_mb,
            vram_used_mb=smi_used_mb,
            vram_free_mb=smi_free_mb,
            utilization_pct=smi_util,
            temperature_celsius=smi_temp,
            power_wattage=smi_power,
            driver_version=driver_ver,
            cuda_runtime_version=cuda_ver,
        )

    def get_disk_telemetry(self) -> DiskTelemetry:
        """Extract filesystem telemetry for the project path."""
        path_str = str(self.project_root.resolve())
        usage = psutil.disk_usage(path_str)
        mount = path_str[:3] if sys.platform == "win32" and len(path_str) >= 3 else "/"

        read_mb = "Unavailable"
        write_mb = "Unavailable"
        try:
            io = psutil.disk_io_counters()
            if io:
                read_mb = round(io.read_bytes / (1024**2), 1)
                write_mb = round(io.write_bytes / (1024**2), 1)
        except Exception:
            pass

        return DiskTelemetry(
            mount_point=mount,
            total_gb=round(usage.total / (1024**3), 2),
            used_gb=round(usage.used / (1024**3), 2),
            free_gb=round(usage.free / (1024**3), 2),
            utilization_pct=round(usage.percent, 1),
            read_mb=read_mb,
            write_mb=write_mb,
            temperature_celsius="Unavailable (Not exposed by OS without storage controller API)",
        )

    def get_process_telemetry(self) -> ProcessTelemetry:
        """Extract resource metrics specific to the current Python application process."""
        p = self._process
        mem_info = p.memory_info()
        cpu_pct = p.cpu_percent(interval=None)

        io_read = "Unavailable"
        io_write = "Unavailable"
        try:
            io = p.io_counters()
            io_read = round(io.read_bytes / (1024**2), 2)
            io_write = round(io.write_bytes / (1024**2), 2)
        except (AttributeError, psutil.AccessDenied):
            pass

        gpu_alloc = "N/A"
        gpu_res = "N/A"
        if torch.cuda.is_available():
            gpu_alloc = round(torch.cuda.memory_allocated(0) / (1024**2), 2)
            gpu_res = round(torch.cuda.memory_reserved(0) / (1024**2), 2)

        return ProcessTelemetry(
            pid=p.pid,
            process_name=p.name(),
            cpu_percent=round(cpu_pct, 1),
            memory_rss_mb=round(mem_info.rss / (1024**2), 1),
            memory_vms_mb=round(mem_info.vms / (1024**2), 1),
            threads_count=p.num_threads(),
            io_read_mb=io_read,
            io_write_mb=io_write,
            gpu_vram_allocated_mb=gpu_alloc,
            gpu_vram_reserved_mb=gpu_res,
        )
