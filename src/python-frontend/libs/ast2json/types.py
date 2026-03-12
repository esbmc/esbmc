import codecs

BUILTIN_PURE = (int, float, bool)
BUILTIN_BYTES = (bytearray, bytes)
BUILTIN_STR = (str)


def decode_str(value):
    return value


def decode_bytes(value):
    try:
        return value.decode('utf-8')
    except Exception:
        return codecs.getencoder('hex_codec')(value)[0].decode('utf-8')
