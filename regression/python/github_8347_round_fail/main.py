# Issue #8347: round() over a call on an unimportable module must stop with the
# conversion error, not report an uncaught TypeError.
import nosuchmodule

n = round(nosuchmodule.f())
