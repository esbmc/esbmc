#include <assert.h>

int main()
{
  unsigned char c = 200;
  unsigned short s = 60000;
  int k = __sync_fetch_and_add(&c, 1);
  int m = __sync_add_and_fetch(&s, 1);
  assert(k == 200 && c == 201);
  assert(m == 60001);
  return 0;
}
