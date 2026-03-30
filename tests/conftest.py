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
# 需要让 torch.Tensor 是一个真正的类，
# 否则 scipy 的 issubclass(x, torch.Tensor) 会崩溃
# ──────────────────────────────────────────────
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:

    class _FakeTensor:
        """占位类，让 issubclass() 检查不崩溃。"""
        pass

    _mock_torch = MagicMock()
    _mock_torch.Tensor = _FakeTensor

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
