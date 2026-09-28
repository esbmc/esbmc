// After setg(data, data, data + 4) the get pointer is at data, so the current
// character is 'a', not 'b' (#7539).
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
  char cur() const
  {
    return *gptr();
  }
};

int main()
{
  Buf b;
  assert(b.cur() == 'b');
  return 0;
}
