# Control for python_class_shadows_model_class: the identical class under a
# name no operational model defines. It verifies, which is what attributes that
# test's wrong verdict to the name collision rather than to the class body.
class MyError:
    def __init__(self) -> None:
        self.v: int = 2

    def get(self) -> int:
        return self.v


o = MyError()
assert o.get() == 2
