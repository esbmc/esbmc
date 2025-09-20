#include <assert.h>

int main()
{
  int x = nondet_int();
  int a = nondet_int();
  int b = nondet_int();

  assert((x && !x) == 0);

  assert(((x && a) && (x && b)) == (x && (a && b)));

  assert(((x || a) || (x || b)) == (x || (a || b)));

  return 0;
}
