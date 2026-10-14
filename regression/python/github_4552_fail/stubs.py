class A:
    def __getitem__(self, key) -> int:
        return key[0].start

class B:
    def __getitem__(self, key) -> int:
        return key[0].start
