# mylib.py - Library with some unsupported features
def advanced_feature(data: str) -> bytes:
    """Uses encoding - not yet supported"""
    return data.encode('utf-8')


def basic_feature(x: int) -> bool:
    """Simple operation - fully supported"""
    return x > 0
