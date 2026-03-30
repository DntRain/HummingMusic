"""
conftest.py - pytest 全局配置

在 CI 环境中 mock 重型依赖（crepe, torch），
本地有完整安装时使用真实模块。
"""

import sys
from unittest.mock import MagicMock

# ──────────────────────────────────────────────
# Mock crepe（CI 不安装）
# ──────────────────────────────────────────────
try:
    import crepe  # noqa: F401
except ImportError:
    sys.modules["crepe"] = MagicMock()

# ──────────────────────────────────────────────
# Mock torch 全家桶（CI 不安装）
# ──────────────────────────────────────────────
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    _mock_torch = MagicMock()
    for mod in [
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "torch.nn.utils",
        "torch.nn.utils.rnn",
        "torch.optim",
        "torch.utils",
        "torch.utils.data",
    ]:
        sys.modules[mod] = _mock_torch
    TORCH_AVAILABLE = False
