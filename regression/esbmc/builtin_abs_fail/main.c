#include <assert.h>

int main()
{
  assert(__builtin_abs(-3) == 3);
  assert(__builtin_labs(-3l) == -3);
  return 0;
}
