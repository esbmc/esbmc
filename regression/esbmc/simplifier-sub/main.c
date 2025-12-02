#include <assert.h>

int main()
{
  int *ptr, offset;
  assert((ptr + offset) - offset == ptr);
  return 0;
}
