#include <assert.h>

int main()
{
  unsigned char c = 200;
  int k = __sync_fetch_and_add(&c, 1);
  assert(k < 128);
  return 0;
}
