// basic_streambuf declared the ten get/put area members and defined none,
// and held no areas, so setg stored nothing and every accessor returned an
// unconstrained pointer -- [streambuf.get.area]'s postconditions did not
// hold after setg, on a buffer the program had just set up (#7539).
#include <cassert>
#include <streambuf>

struct Buf : std::streambuf
{
  char data[4];
  Buf()
  {
    data[0] = 'a';
    data[1] = 'b';
    data[2] = 'c';
    data[3] = 'd';
    setg(data, data, data + 4);
  }
  bool areas_set() const
  {
    return eback() == data && gptr() == data && egptr() == data + 4;
  }
  bool ordered() const
  {
    return eback() <= gptr() && gptr() <= egptr();
  }
  char cur() const
  {
    return *gptr();
  }
  void advance()
  {
    gbump(1);
  }
};

int main()
{
  Buf b;
  assert(b.areas_set());
  assert(b.ordered());
  assert(b.cur() == 'a');
  b.advance();
  assert(b.cur() == 'b');
  assert(b.ordered());
  return 0;
}
