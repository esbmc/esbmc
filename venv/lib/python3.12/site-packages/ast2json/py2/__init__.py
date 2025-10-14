import codecs

BUILTIN_PURE = (int, float, bool, long)
BUILTIN_BYTES = (bytearray, bytes)
BUILTIN_STR = (str, basestring)


def decode_str(value):
    try:
        return value.decode('utf-8')
    except Exception:
        return codecs.getencoder('hex_codec')(value)[0].decode('utf-8')


def decode_bytes(value):
    try:
        return value.decode('utf-8')
    except Exception:
        return codecs.getencoder('hex_codec')(value)[0].decode('utf-8')
