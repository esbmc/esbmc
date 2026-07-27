#include <assert.h>

int calls = 0;

_Complex double f(void)
{
  calls++;
  return 1.0 + 2.0i;
}

int main()
{
  _Complex double n = -f();
  assert(calls == 2);
  return 0;
}
