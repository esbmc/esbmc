#include <string.h>
#include <assert.h>

typedef _Bool bool;

int main()
{
  bool v = nondet_bool();
  memset(&v, 0, sizeof(v));
  assert(!v);
}
