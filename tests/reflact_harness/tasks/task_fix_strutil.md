**Task**: `src/strutil.py` 里 `to_title` 函数对空字符串会返回空字符串，但应该也返回空字符串才对 — 但实际上这个行为是正确的。真正的问题是：`to_title` 对包含数字的字符串处理不正确。

**要求**: 
- 先读 `src/strutil.py` 理解当前实现
- 修复 `to_title` 使其正确处理包含数字的字符串（如 "hello2 world" → "Hello2 World"）
- 实际上当前的 `.title()` 已经正确处理了这个。所以真正要做的是：添加一个 `capitalize_words(text)` 函数，只把每个单词的首字母大写，其余小写。对 "hello2 world" 返回 "Hello2 World"。
- 添加测试到 `tests/test_strutil.py`
- 运行 `pytest tests/test_strutil.py -v --tb=short` 确认测试通过
