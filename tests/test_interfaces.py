"""
test_interfaces.py - 接口定义模块测试

验证接口函数签名和类型约束。
"""


class TestInterfaceDefinitions:
    """接口定义正确性测试。"""

    def test_valid_styles(self):
        """VALID_STYLES 应包含四种风格。"""
        from src.interfaces import VALID_STYLES

        assert set(VALID_STYLES) == {"pop", "jazz", "classical", "folk"}

    def test_interface_functions_importable(self):
        """四个核心接口函数应可正常导入。"""
        from src.interfaces import (
            extract_pitch,
            quantize_humming,
            render_audio,
            transfer_style,
        )

        assert callable(extract_pitch)
        assert callable(quantize_humming)
        assert callable(transfer_style)
        assert callable(render_audio)

    def test_protocol_classes_importable(self):
        """协议类应可正常导入。"""
        from src.interfaces import (  # noqa: F401
            AudioRenderer,
            HummingQuantizer,
            PitchExtractor,
            StyleTransferEngine,
        )
