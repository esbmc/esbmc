#include <assert.h>
void f(void)
{
  int *p = __builtin_alloca(sizeof(int));
  *p = 1;
  assert(*p == 1);
}
int main(void)
{
  int *q;
  {
    q = __builtin_alloca(sizeof(int));
    *q = 2;
  }
  f();
  assert(*q == 2);
  return 0;
}
