#include <fstream>
#include <iostream>
#include <cassert>

typedef std::basic_ifstream<char, std::char_traits<char>> IFStream;
typedef std::basic_ofstream<char, std::char_traits<char>> OFStream;
typedef std::basic_fstream<char, std::char_traits<char>> FStream;
typedef std::basic_iostream<char, std::char_traits<char>> IOStream;

int main()
{
  IFStream in;
  OFStream out;
  FStream both;

  assert(!in.is_open());
  assert(!out.is_open());
  assert(!both.is_open());

  std::ifstream *p = &in;
  std::ofstream *q = &out;
  std::fstream *r = &both;
  assert(p == &in);
  assert(q == &out);
  assert(r == &both);

  IOStream *io = 0;
  std::iostream *s = io;
  assert(s == 0);

  return 0;
}
