#include <assert.h>

int main()
{
  _Bool v = 1;
  char *ptr = (char *)&v;
  ptr[0] = 0;
  assert(v);
  return 0;
}
