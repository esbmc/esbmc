# KNOWNBUG (unsound false-SUCCESSFUL): a bytes method result is mis-folded to
# str, so an (always-False in CPython) bytes-vs-str comparison verifies. In
# CPython b"abc".upper() is b"ABC" (bytes), and b"ABC" == "ABC" is False, so
# this assertion must FAIL. ESBMC proves it because python_consteval's
# Constant handler (python_consteval.cpp:728) folds a bytes literal to a
# STRING (the real bytes live in `encoded_bytes` / the node is tagged
# `esbmc_type_annotation: "bytes"`, both ignored), so the .upper() fold
# produces a str. A one-line "decline bytes in the Constant handler" fix would
# break the ~260 bytes tests that depend on that same fold; the sound fix
# needs a dedicated BYTES kind in PyConstValue so bytes fold as bytes.
assert b"abc".upper() == "ABC"
