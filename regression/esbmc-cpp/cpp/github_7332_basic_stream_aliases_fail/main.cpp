#include <fstream>
#include <cassert>

typedef std::basic_ifstream<char, std::char_traits<char>> IFStream;

int main()
{
  IFStream in;
  assert(in.is_open());
  return 0;
}
