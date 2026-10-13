# Issue #8347: abs() over a call on an unimportable module must stop with the
# conversion error, not report an uncaught TypeError.
import nosuchmodule

n = abs(nosuchmodule.f())
