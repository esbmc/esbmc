#include <assert.h>
#include <limits.h>

typedef unsigned v4 __attribute__((vector_size(16)));

int main()
{
  int x = nondet_int();

  assert(x - x == 0);

  assert((~x & x) == 0);
  assert((x & ~x) == 0);

  assert((0 & x) == 0);
  assert((x & 0) == 0);
  assert((-1 & x) == x);
  assert((x & -1) == x);

  assert((~x | x) == -1);
  assert((x | ~x) == -1);

  assert((x | 0) == x);
  assert((x | -1) == -1);
  assert((0 | x) == x);
  assert((-1 | x) == -1);

  assert((x ^ x) == 0);
  assert((x ^ 0) == x); 

  assert(~(~x) == x);

  unsigned y = nondet_uint();

  assert((y < 0) == 0);

  unsigned u = nondet_uint();
  int s = nondet_int();

  // unsigned simplification: u < 0 should always be false
  assert((u < 0) == 0);

  // signed case: s < 0 is *not* always false
  if (s < 0) assert(1); // must be satisfiable

  unsigned char c = nondet_uchar();

  // -1 in unsigned char means 255 (all bits set)
  assert((c & (unsigned char)-1) == c);
  assert((c | (unsigned char)-1) == (unsigned char)255);

  // double negation on small width
  assert((unsigned char)~((unsigned char)~c) == c);

  // Check x & -1 for unsigned
  assert((u & (unsigned)-1) == u);

  assert((s & -1) == s);  // OK
  assert((u & -1) == u);  // OK (unsigned all bits set)

  v4 a = {1, 2, 3, 4};
  v4 b = {0, 0, 0, 0};

  // Simplification should hold element-wise
  assert((a & b)[0] == 0);
  assert((a | b)[1] == 2);
  assert((a ^ a)[2] == 0);

  // Multiple reductions in one expression
  assert(((x ^ x) | (x & -1)) == x);

  // ~(~(~x)) = ~x
  assert(~(~(~x)) == ~x);

  // Mix AND/OR/NOT with constants
  assert(((x & 0) | ~x) == ~x);

  assert((1 * x) == x);

  x = INT_MIN;
  assert(x - x == 0);

  return 0;
}
