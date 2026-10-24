#include <assert.h>

int global __attribute__((weak)) = 999;
int weak_only __attribute__((weak)) = 7;
__attribute__((weak)) int combine(int a, int b)
{
  return a + b;
}

int main(void)
{
  assert(global == 1234);
  assert(weak_only == 7);
  assert(combine(5, 3) == 2);
  return 0;
}
