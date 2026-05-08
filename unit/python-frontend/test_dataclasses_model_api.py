import importlib.util
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "src", "python-frontend", "models", "dataclasses.py")


spec = importlib.util.spec_from_file_location("esbmc_dataclasses_model", MODEL_PATH)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def test_is_dataclass_true_for_class_and_instance():
    @model.dataclass
    class C:
        x: int
        def __init__(self, x):
            self.x = x

    assert model.is_dataclass(C) is True
    assert model.is_dataclass(C(1)) is True


def test_fields_excludes_classvar_and_initvar():
    class ClassVar:
        def __class_getitem__(cls, item):
            return cls

    @model.dataclass
    class C:
        x: int
        y: ClassVar[int]
        z: model.InitVar[int]
        def __init__(self, x):
            self.x = x

    names = [f.name for f in model.fields(C)]
    assert names == ["x"]


def test_asdict_and_astuple_recursive():
    @model.dataclass
    class Child:
        v: int
        def __init__(self, v):
            self.v = v

    @model.dataclass
    class Parent:
        c: Child
        xs: list
        m: dict
        def __init__(self, c, xs, m):
            self.c = c
            self.xs = xs
            self.m = m

    p = Parent(Child(7), [Child(1), Child(2)], {"k": Child(3)})

    assert model.asdict(p) == {
        "c": {"v": 7},
        "xs": [{"v": 1}, {"v": 2}],
        "m": {"k": {"v": 3}},
    }
    assert model.astuple(p) == ((7,), [(1,), (2,)], {"k": (3,)})


def test_replace_updates_only_specified_fields():
    @model.dataclass
    class C:
        x: int
        y: int
        def __init__(self, x, y):
            self.x = x
            self.y = y

    c1 = C(1, 2)
    c2 = model.replace(c1, y=9)

    assert c1.x == 1 and c1.y == 2
    assert c2.x == 1 and c2.y == 9


def test_replace_rejects_unknown_field():
    @model.dataclass
    class C:
        x: int
        def __init__(self, x):
            self.x = x

    try:
        model.replace(C(1), y=2)
        assert False
    except TypeError as exc:
        assert "unexpected field name" in str(exc)
