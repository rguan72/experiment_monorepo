# Code Style Preferences

## Core Principles

- **Minimal code**: Keep code concise and avoid verbosity
- **No unnecessary logging**: Consolidate log statements where possible
- **No defensive try-catch blocks**: Only add error handling when explicitly needed
- **Compact formatting**: Prefer single expressions over multi-line loops when readable

## Examples

### Logging

**Avoid**: Multiple separate logger.info() calls with decorative separators
```python
logger.info("=" * 80)
logger.info("📤 OUTGOING REQUEST")
logger.info(f"Timestamp: {datetime.now().isoformat()}")
logger.info(f"Method: {method}")
logger.info(f"Path: {path}")
logger.info("-" * 80)
logger.info("Headers:")
for key, value in headers.items():
    if key.lower() == "authorization":
        logger.info(f"  {key}: {value[:20]}..." if len(value) > 20 else f"  {key}: ***")
    else:
        logger.info(f"  {key}: {value}")
```

**Prefer**: Consolidated logging with string formatting
```python
logger.info(f"{'=' * 80}\n📤 REQUEST: {method} {path} @ {datetime.now().isoformat()}\n{'-' * 80}")
logger.info("Headers:\n" + "\n".join(
    f"  {k}: {v[:20]}..." if k.lower() == "authorization" and len(v) > 20 else f"  {k}: {v}"
    for k, v in headers.items()
))
```

### Error Handling

**Avoid**: Unnecessary try-catch blocks
```python
try:
    body_json = json.loads(body)
    logger.info("Body:")
    logger.info(json.dumps(body_json, indent=2))
except json.JSONDecodeError:
    logger.info(f"Body (raw): {body[:500]}...")
```

**Prefer**: Let errors propagate or handle only when necessary; extract to helper functions
```python
logger.info(f"Body:\n{_format_body(body)}")
```

## General Guidelines

- Use list comprehensions and generator expressions instead of explicit loops
- Combine related operations into single statements
- Extract repeated logic to helper functions rather than adding verbosity
- Avoid decorative comments and excessive whitespace
- Don't add error handling for scenarios that shouldn't fail in normal operation
