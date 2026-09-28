#include <cassert>

extern "C" int nondet_int();

// A conditional over two arrays decays per arm: `c ? "ab" : "xy"` passed as a
// `const char *` is `c ? &"ab"[0] : &"xy"[0]`. Indexing the conditional instead
// reaches compute_pointer_offset as an `if` it cannot read, and ESBMC aborts.
//
// The arms must be the *same length*: with different lengths each decays on its
// own and no array-typed conditional is ever formed, so the test would pin
// nothing.
static int length(const char *s)
{
  int n = 0;
  while (s[n] != '\0')
    n++;
  return n;
}

int main()
{
  int i = nondet_int();
  assert(length(i < 1 ? "ab" : "xy") == 2);
}
