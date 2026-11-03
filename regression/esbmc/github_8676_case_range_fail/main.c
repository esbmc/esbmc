#include <assert.h>
int nondet_int(void);
int main(void)
{
  int v = nondet_int();
  switch (v)
  {
  case 1 ... 5:
    assert(v != 3);
    break;
  }
  return 0;
}
