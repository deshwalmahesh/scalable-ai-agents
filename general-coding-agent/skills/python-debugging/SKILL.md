---
name: python-debugging
description: Instructions and heuristics for diagnosing and debugging Python runtime exceptions, stack traces, and test regressions.
---

# Python Debugging Skill Guide

When diagnosing or fixing a Python bug:
1. **Locate the Origin**: Examine the deepest frame of the traceback first to identify the exact file and line number where the exception was raised.
2. **Inspect Context**: Use `fs_read` to read the surrounding 20 lines of code around the failure.
3. **Trace Root Cause**: Do not apply superficial patches or catch generic exceptions. Diagnose why invalid arguments, types, or None values were passed.
4. **Minimal Patching**: Use `fs_edit` to apply the targeted fix atomically.
5. **Verify**: Run the corresponding test or execution command with `shell_exec` to ensure the fix resolves the issue without regressions.
