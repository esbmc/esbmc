#include <cassert>

int main()
{
  char8_t c = u8'A';
  // 'A' is 0x41 = 65; the assertion below must fail.
  assert(c == 66);
  return 0;
}
