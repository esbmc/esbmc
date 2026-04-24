# Pins current ESBMC behavior for ``field(default_factory=...)`` when the
# caller passes the same field as a kwarg at construction time.
#
# Marco D limitation (tracked for Marco F): factory fields are NOT exposed as
# parameters of the synthesized ``__init__`` to side-step a converter
# limitation on Call expressions in parameter defaults. The factory call is
# emitted directly in the body, guaranteeing per-instance fresh values
# (sound for mutable factories) but silently ignoring caller-supplied
# overrides like ``C(x=200)``. CPython would honor the override.
#
# This regression test pins the current behavior so any Marco F change that
# allows the override intentionally breaks it and forces an update.
from dataclasses import dataclass, field


def make() -> int:
    return 100


@dataclass
class C:
    x: int = field(default_factory=make)


c = C(x=200)
# Factory wins; kwarg silently dropped.
assert c.x == 100

