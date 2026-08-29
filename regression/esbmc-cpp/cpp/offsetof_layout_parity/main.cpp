// C++ offsetof now lowers through clang's OffsetOfExpr (clang_c_convert.cpp)
// rather than the __ESBMC_POINTER_OFFSET expansion C still uses. Parity guard:
// the two have to agree on every layout, or the language a shared header is
// compiled as would change the offsets it computes. Packed, over-aligned,
// nested, union and array-indexed members are in because that is where two
// layout models diverge if they diverge at all.
#include <cstddef>
#include <cassert>

#define ESB(t, m) ((std::size_t)__ESBMC_POINTER_OFFSET(&((t *)0)->m))
#define CLG(t, m) offsetof(t, m)
#define BOTH(t, m) assert(ESB(t, m) == CLG(t, m))

struct plain
{
  char a;
  int b;
  double c;
};

struct mixed
{
  char a;
  char b;
  short c;
  long d;
};

#pragma pack(push, 1)
struct packed
{
  char a;
  int b;
  double c;
};
#pragma pack(pop)

struct nested
{
  struct plain x;
  struct mixed y;
  char z;
};

struct overaligned
{
  char a;
  __attribute__((aligned(16))) int b;
  char c;
};

union un
{
  char a;
  double b;
  int c;
};

struct with_array
{
  char a;
  int b[7];
  char c;
};

int main(void)
{
  BOTH(struct plain, a);
  BOTH(struct plain, b);
  BOTH(struct plain, c);

  BOTH(struct mixed, a);
  BOTH(struct mixed, b);
  BOTH(struct mixed, c);
  BOTH(struct mixed, d);

  BOTH(struct packed, a);
  BOTH(struct packed, b);
  BOTH(struct packed, c);

  BOTH(struct nested, x);
  BOTH(struct nested, y);
  BOTH(struct nested, z);
  BOTH(struct nested, x.c);
  BOTH(struct nested, y.d);

  BOTH(struct overaligned, a);
  BOTH(struct overaligned, b);
  BOTH(struct overaligned, c);

  BOTH(union un, a);
  BOTH(union un, b);
  BOTH(union un, c);

  BOTH(struct with_array, a);
  BOTH(struct with_array, b);
  BOTH(struct with_array, c);
  BOTH(struct with_array, b[3]);
  return 0;
}
