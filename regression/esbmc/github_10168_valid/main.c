#include <assert.h>

int main(void)
{
  int x = 2;
  int *p = &x;
  switch (*p)
  {
  default:
    x = 3;
  }
  switch (*p)
  {
  }
  assert(x == 3);
  return 0;
}
