#include <assert.h>

/* Regression test for the add_padding idempotency fix in
 * clang-c-frontend/padding.cpp. Prior to the fix, a struct
 * containing an _ExtInt member visited by the type adjuster more
 * than once (e.g. via the debug assertion in
 * clang_c_adjust_expr.cpp, or in Release via multiple expressions
 * sharing the same struct tag) would accumulate duplicate
 * `ext_int_pad$N` padding components with colliding names, and
 * later member lookups aborted with:
 *   Name "ext_int_pad$N" matches more than one member in
 *   struct/union "struct Foo"
 * (irep2_type.cpp). The fix skips pad_ext_int_after when the
 * component is already followed by an extint padding.
 *
 * The trigger below uses an _ExtInt member whose type is shared
 * across many expressions (struct copies, by-value returns,
 * pointer dereferences, field accesses) to maximise the number
 * of type-adjustment passes over the struct. */

typedef unsigned _ExtInt(256) u256;

struct S {
  u256 data;
  unsigned long length;
};

static struct S make(u256 v, unsigned long n)
{
  struct S s;
  s.data = v;
  s.length = n;
  return s;
}

static int eq(const struct S *a, const struct S *b)
{
  if (a->length != b->length) return 0;
  return a->data == b->data;
}

int main(void)
{
  struct S a = make((u256)0xBEEF, 32);
  struct S b = make((u256)0xBEEF, 32);
  struct S c = make((u256)0xDEAD, 32);
  assert(eq(&a, &b));
  assert(!eq(&a, &c));
  /* Field access on a by-value copy — another adjustment pass. */
  struct S d = a;
  assert(d.data == (u256)0xBEEF);
  assert(d.length == 32);
  return 0;
}
