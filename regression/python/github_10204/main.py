# Issue #10204: a float local passed through self to an unannotated parameter.
class M:
    def get_time(self, tnow):
        return int(tnow - 1.5)

    def idle(self):
        now: float = 0.0
        w = self.get_time(now)
        assert w == -1


M().idle()
