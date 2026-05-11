#include <cassert>

int main()
{
  char8_t c = u8'A';
  assert(c == 65);

  char8_t s[] = u8"hi";
  assert(s[0] == 0x68);
  assert(s[1] == 0x69);
  assert(s[2] == 0);

  return 0;
}
