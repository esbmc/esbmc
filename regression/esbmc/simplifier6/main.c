#include <assert.h>

int main() 
{
  _Bool x = nondet_bool();
  _Bool y = nondet_bool();

  assert((~(x & y)) == ((~x) | (~y)));
  assert((~(x | y)) == ((~x) & (~y)));
  assert((x & (x | y)) == x);
  assert(x == (x & (x | y)));
  assert((x | (x & y)) == x);
  assert(x == (x | (x & y)));
  
  int a = nondet_int();

  assert((a >> 0) == a);
  assert((x >> 0) == x);

  int b = nondet_int();

  assert((a && (a || b)) == (_Bool)a);
  assert((_Bool)a == (a && (a || b)));

  assert((a || (a && b)) == (_Bool)a);
  assert((_Bool)a == (a || (a && b)));

  assert((!!x) == x);

  assert((!(a && b)) == (!a || !b));
  assert((!a || !b) == (!(a && b)));

  assert((!(a || b)) == (!a && !b));

  assert((!(a == b)) == (a != b));
  assert((!(a != b)) == (a == b));
  assert((!(a < b)) == (a >= b));
  assert((!(a <= b)) == (a > b));
  assert((!(a > b)) == (a <= b));
  assert((!(a >= b)) == (a < b));

  int c = nondet_int();
  int d = nondet_int();

  unsigned u = nondet_uint();
  assert((u >= 0) == 1);

  return 0;
}

