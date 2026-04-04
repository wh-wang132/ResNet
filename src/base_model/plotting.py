#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目统一的 Matplotlib 配置。"""

import matplotlib


_MATPLOTLIB_CONFIGURED = False


def configure_matplotlib():
    """配置无头后端、字体和负号显示规则。"""
    global _MATPLOTLIB_CONFIGURED

    if not _MATPLOTLIB_CONFIGURED:
        matplotlib.use("Agg")
        _MATPLOTLIB_CONFIGURED = True

    matplotlib.rcParams["font.sans-serif"] = ["Times New Roman"]
    matplotlib.rcParams["axes.unicode_minus"] = False
