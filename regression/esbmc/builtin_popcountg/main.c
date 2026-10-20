#include <assert.h>
int main(void)
{
  unsigned char c = 0xff;
  unsigned short s = 0x8001;
  unsigned __int128 w = ((unsigned __int128)1 << 100) | 7;
  assert(__builtin_popcountg(c) == 8);
  assert(__builtin_popcountg(s) == 2);
  assert(__builtin_popcountg(w) == 4);
  assert(__builtin_popcountg(0ull) == 0);
  return 0;
}
