#include <cassert>
#include <string>

int main()
{
  // n == capacity() is the largest resize this fixed-capacity model admits;
  // the terminator at str[n] must still land inside the buffer.
  std::string s;
  s.resize(127);
  assert(s.size() == 127);
  assert(s.c_str()[127] == '\0');

  return 0;
}
