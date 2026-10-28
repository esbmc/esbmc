class _Proxy:

    def __getattr__(self, name):
        raise RuntimeError("Working outside of request context.")


request = _Proxy()
