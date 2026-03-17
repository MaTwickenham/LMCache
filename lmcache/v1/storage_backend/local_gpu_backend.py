# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import GPUMemoryAllocator, MemoryAllocatorInterface
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)


class LocalGPUBackend(LocalCPUBackend):
    """GPU-resident hot cache backend.

    This backend intentionally reuses the existing hot-cache/key/pin/policy
    mechanics from LocalCPUBackend so we can introduce a GPU tier without
    rewriting the cache controller path up front.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheMetadata] = None,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        memory_allocator: Optional[MemoryAllocatorInterface] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("LocalGPUBackend requires CUDA")
        super().__init__(
            config,
            metadata=metadata,
            dst_device=dst_device,
            lmcache_worker=lmcache_worker,
            memory_allocator=memory_allocator,
        )
        self.use_hot = config.local_gpu

    def _setup_metrics(self):
        # Reusing LocalCPUBackend's Prometheus metrics would overwrite the CPU
        # counters with GPU values. Leave GPU metrics to a follow-up patch.
        return

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: Optional[LMCacheMetadata] = None,
    ) -> MemoryAllocatorInterface:
        gpu_size = config.max_local_gpu_size

        if metadata is not None:
            save_only_first_rank = (
                config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
                and metadata.use_mla
            )
            if save_only_first_rank and metadata.is_first_rank():
                gpu_size = config.get_extra_config_value(
                    "first_rank_max_local_gpu_size",
                    gpu_size,
                )

        gpu_size_bytes = int(gpu_size * 1024**3)
        logger.info(
            "Initializing LocalGPUBackend allocator on %s with %.2f GB",
            self.dst_device,
            gpu_size,
        )
        return GPUMemoryAllocator(gpu_size_bytes, device=self.dst_device)
