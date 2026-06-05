**Task**: 用户反馈 `src/calc.py` 里的 `divide` 函数除以零返回 `None` 不够好。应该返回一个有意义的结果。

**要求**:
- 先读 `src/calc.py` 和 `tests/test_calc.py`
- 修改 `divide` 使其除以零时返回 `float('inf')` 而不是 `None`
- 对负除以零返回 `float('-inf')`
- 更新 `tests/test_calc.py` 中的 `test_divide_by_zero`
- 运行 `pytest tests/test_calc.py -v --tb=short` 确认测试通过
