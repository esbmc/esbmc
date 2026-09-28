// github #6304: a bool bitfield used in a boolean context (&&, ||, !, if,
// assertion) previously aborted ESBMC with an internal type assertion, because
// the bitfield migrated to unsignedbv[1] but its legacy type stayed `bool`, so
// the boolean-context coercion inserted no cast. Basing the bitfield on an
// unsigned bitvector (as integer bitfields already are) fixes it.
#include <cassert>

struct F
{
  bool a : 1, b : 1, c : 1;
};

int main()
{
  F f{};
  f.a = 1;
  f.c = 1;

  assert(f.a && !f.b && f.c);   // &&, !
  assert(f.a || f.b);           // ||
  if (f.a)                      // if
    assert(!f.b);

  bool x = f.a;                 // read into a bool
  assert(x);
  f.b = f.a;                    // bitfield-to-bitfield
  assert(f.b);
  return 0;
}
