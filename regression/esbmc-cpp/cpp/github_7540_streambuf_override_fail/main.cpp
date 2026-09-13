// Counterpart of github_7540_streambuf_override: the override returns the
// 64-bit offset it was given, so asserting a truncated value fails (#7540).
#include <streambuf>

#include <cassert>
#include <ios>

class Buf : public std::streambuf
{
public:
  std::streampos seek(std::streamoff off)
  {
    return seekoff(off, std::ios_base::beg, std::ios_base::in);
  }

protected:
  std::streampos seekoff(
    std::streamoff off,
    std::ios_base::seekdir,
    std::ios_base::openmode) override
  {
    return off;
  }
};

int main()
{
  Buf b;
  std::streampos p = b.seek(3000000000LL);
  assert(p < std::streampos(0));
  return 0;
}
