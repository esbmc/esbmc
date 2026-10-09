#include <assert.h>

int main()
{
  assert(__builtin_rotateleft32(0x80000001u, 1) == 3);
  assert(__builtin_rotateright32(0x80000001u, 1) == 3);
  return 0;
}
