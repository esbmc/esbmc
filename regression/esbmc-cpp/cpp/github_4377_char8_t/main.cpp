// esbmc/esbmc#4377 (C++20): char8_t used to abort conversion with
// "Unrecognized clang builtin type char8_t".
#include <cassert>

int main()
{
  char8_t c = u8'A';
  assert(c == 65);
  assert(sizeof(char8_t) == 1);

  const char8_t *s = u8"AB";
  assert(s[0] == 65);
  assert(s[1] == 66);
  assert(s[2] == 0);
  return 0;
}
