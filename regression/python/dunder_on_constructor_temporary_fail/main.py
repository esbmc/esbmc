class Sized:
    def __len__(self):
        return 3


# len(Sized()) calls __len__, which returns 3.
assert len(Sized()) == 4
