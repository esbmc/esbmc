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

  /* The conversions pin alias identity -- a distinct class would not convert --
   * and is_open() then asserts through the alias-typed object. */
  std::ifstream *p = &in;
  std::ofstream *q = &out;
  std::fstream *r = &both;
  assert(!p->is_open());
  assert(!q->is_open());
  assert(!r->is_open());

  IOStream *io = 0;
  std::iostream *s = io;
  assert(s == 0);

  return 0;
}
