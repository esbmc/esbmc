# Corrupt goto binary: forward irep reference

`forwardref.goto` is a hand-built 25-byte goto binary whose single symbol record
claims irep id 7 while the reader's back-reference table is still empty, then
supplies a complete irep body so deserialisation of that body succeeds.

Write-side ids are dense and increasing (`reference_convert()` numbers each new
irep with `ireps_on_write.size()`), so id 7 against an empty table cannot occur
in a well-formed stream. The reader rejects it. Without that check the id would
reach an unchecked `std::vector::operator[]` write at a file-controlled offset.

Regenerate with:

    python3 -c 'import struct; L=lambda u: struct.pack(">I",u); \
      open("forwardref.goto","wb").write(b"GBF"+L(1)+L(1)+L(7)+L(0)+b"root\0"+b"\0")'

The trailing `L(0) + b"root\0" + b"\0"` is what makes this fixture sharper than a
truncated file: it keeps `read_irep()` from failing first, so the test exercises
the bounds check rather than an unrelated end-of-stream path.
