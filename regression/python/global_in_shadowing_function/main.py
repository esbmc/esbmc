# A name used inside a function must resolve in that function's own scope, not
# be looked up in an imported module under the enclosing function's name. Here
# `acos` collides with math.acos, which made HALF_TURN resolve to &math.acos --
# a function pointer in arithmetic, breaking every solver backend (#6895).
import geom

assert geom.acos(80.0) == 100.0
