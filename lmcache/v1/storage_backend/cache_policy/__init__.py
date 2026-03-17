# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Dict, Type
import importlib
import inspect

# First Party
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy
from lmcache.v1.storage_backend.cache_policy.fifo import FIFOCachePolicy
from lmcache.v1.storage_backend.cache_policy.lfu import LFUCachePolicy
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy
from lmcache.v1.storage_backend.cache_policy.mru import MRUCachePolicy

# Cache policy mapping
POLICY_MAPPING: Dict[str, Type[BaseCachePolicy]] = {
    "LRU": LRUCachePolicy,
    "LFU": LFUCachePolicy,
    "FIFO": FIFOCachePolicy,
    "MRU": MRUCachePolicy,
}


def _resolve_policy_class(policy_name: str) -> Type[BaseCachePolicy]:
    upper_policy_name = policy_name.upper()
    if upper_policy_name in POLICY_MAPPING:
        return POLICY_MAPPING[upper_policy_name]

    if ":" in policy_name:
        module_name, class_name = policy_name.split(":", maxsplit=1)
    elif "." in policy_name:
        module_name, class_name = policy_name.rsplit(".", maxsplit=1)
    else:
        raise ValueError(
            f"Unknown cache policy: {policy_name}. "
            f"Supported built-ins are: {list(POLICY_MAPPING.keys())}. "
            "Custom policies should use 'module.submodule.ClassName' or "
            "'module.submodule:ClassName'."
        )

    module = importlib.import_module(module_name)
    policy_cls = getattr(module, class_name)
    if not inspect.isclass(policy_cls):
        raise TypeError(
            f"Resolved cache policy '{policy_name}' to non-class object: {policy_cls!r}"
        )
    return policy_cls


def _instantiate_policy(
    policy_cls: Type[BaseCachePolicy],
    **kwargs: Any,
) -> BaseCachePolicy:
    signature = inspect.signature(policy_cls)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return policy_cls(**kwargs)

    accepted_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return policy_cls(**accepted_kwargs)


def get_cache_policy(policy_name: str, **kwargs: Any) -> BaseCachePolicy:
    """
    Factory function to get the cache policy instance based on the policy name.

    Args:
        policy_name: Name of the cache policy (case-insensitive, e.g., "LRU", "lru").

    Returns:
        Instance of the corresponding cache policy.

    Raises:
        ValueError: If the policy name is not supported.
    """
    if not policy_name:
        raise ValueError("Cache policy name cannot be empty")

    policy_cls = _resolve_policy_class(policy_name)
    return _instantiate_policy(policy_cls, **kwargs)
