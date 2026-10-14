class Buffer:
    def __init__(self):
        self.data = [1, 2, 3]

    def ravel(self):
        return self.data


# A user class with its own ravel() method is unrelated to numpy's .flat.
# Writing through its result is still unsupported (Buffer.ravel() returns a
# fresh reference, not something extract_target_name can resolve to a
# symbol), but it must not be reported as the numpy-specific .flat
# diagnostic, which only applies to a numpy-sourced receiver.
b = Buffer()
b.ravel()[0] = 99
assert b.data[0] == 99
