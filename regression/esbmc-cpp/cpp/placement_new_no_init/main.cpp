// Placement new with no initializer: `new (p) T;` for a non-class T.
// Default-initializing a non-class type performs no initialization
// ([dcl.init.general]), so the expression is just the placement address.
// Clang attaches no initializer child to such a CXXNewExpr, which used to
// yield a one-operand "comma" and an out-of-bounds read in adjust_comma.
// See esbmc/esbmc#6184.
#include <new>
#include <cassert>

int main()
{
  // Scalar, no initializer.
  alignas(int) char ibuf[sizeof(int)];
  int *p = new (ibuf) int;
  assert((void *)p == (void *)ibuf); // result aliases the placement address
  *p = 42;                           // default-init leaves it indeterminate
  assert(*p == 42);

  // unsigned char: the aws-sdk-cpp ByteBuffer element type from #6184.
  unsigned char cbuf[1];
  unsigned char *q = new (cbuf) unsigned char;
  *q = 7;
  assert(*q == 7);

  // The Aws::NewArray shape: `new (base + i) T;` in a loop.
  alignas(int) char abuf[3 * sizeof(int)];
  int *base = (int *)abuf;
  for (int i = 0; i < 3; ++i)
  {
    int *e = new (base + i) int;
    *e = i;
  }
  assert(base[0] == 0);
  assert(base[1] == 1);
  assert(base[2] == 2);

  return 0;
}
