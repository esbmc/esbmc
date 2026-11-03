#include <assert.h>

__attribute__((weak)) int combine(int a, int b)
{
  return a + b;
}

int main(void)
{
  assert(combine(5, 3) == 8);
  return 0;
}
