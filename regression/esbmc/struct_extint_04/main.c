#include <assert.h>

/* Smoke test for a struct containing a `_ExtInt(256)` inside a 2D
 * fixed array of another struct. Before the zero-width-padding
 * guard in clang-c-frontend/padding.cpp (paired with the
 * idempotency fix that shipped in struct_extint_03's commit), any
 * struct containing `_ExtInt(N)` where N is already aligned to the
 * representation boundary (N == repr_bits, e.g. 256) accumulated
 * a spurious zero-width `ext_int_pad$N` component. The component
 * leaks into symex's assign field-walking.
 *
 * Under the pure C frontend, symex happens to route these struct
 * writes through byte-extract handlers that tolerate the extra
 * zero-width slot — so THIS particular test passes either way.
 * It's a trip-wire for a FUTURE regression: if some later change
 * makes the C assign pipeline field-walk such structs directly
 * (same pattern as the Solidity frontend already does), the
 * zero-width slot would re-surface as "assignment to constant_int
 * not handled" at src/goto-symex/symex_assign.cpp:448. The
 * intended real-world regression is the Solidity-frontend test
 * `regression/esbmc-solidity/multi_dim_fixed_bytes32_2d_fail`
 * under an L2 uint256-backed BytesStatic model; that test crashed
 * without the zero-width-padding guard and passes with it. */

typedef unsigned _ExtInt(256) u256;

/* Inner struct: _ExtInt(256) + size_t. Before the fix, this would
 * get a zero-width ext_int_pad$1 component between `data` and
 * `length`. */
struct Inner {
  u256 data;
  unsigned long length;
};

/* Outer struct: 2D array of Inner. The nested write
 * `outer->grid[i][j] = v` forces symex to field-walk Inner
 * through nested index + member, which exposes the zero-width
 * component. */
struct Outer {
  struct Inner grid[2][3];
};

static struct Outer g;

static struct Inner make(u256 v, unsigned long n)
{
  struct Inner s;
  s.data = v;
  s.length = n;
  return s;
}

int main(void)
{
  /* The assignment below is the canonical pattern that used to
   * abort with "assignment to constant_int not handled". */
  g.grid[0][0] = make((u256)0x1111, 32);
  g.grid[1][2] = make((u256)0xBEEF, 32);

  /* Readback, field access through the 2D chain. */
  assert(g.grid[0][0].data == (u256)0x1111);
  assert(g.grid[0][0].length == 32);
  assert(g.grid[1][2].data == (u256)0xBEEF);
  assert(g.grid[1][2].length == 32);

  /* Untouched slots should be zero-initialised (static storage). */
  assert(g.grid[0][1].data == (u256)0);
  assert(g.grid[0][2].length == 0);
  return 0;
}
