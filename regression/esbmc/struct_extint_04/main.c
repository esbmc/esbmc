#include <assert.h>

/* Nested aggregate assignment and zero initialization with wide integers.
 * The padding unit tests check the absence of zero-width IR components. */

typedef unsigned _ExtInt(256) u256;

/* Inner struct: _ExtInt(256) + size_t. Before the fix, this would
 * get a zero-width ext_int_pad$1 component between `data` and
 * `length`. */
struct Inner
{
  u256 data;
  unsigned long length;
};

struct Outer
{
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
