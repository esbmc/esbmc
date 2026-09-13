// A streambuf override spelled with the standard position types must match
// basic_streambuf's virtuals, which now use the same 64-bit types (#7540).
#include <cassert>
#include <ios>
#include <streambuf>

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
  assert(b.seek(3000000000LL) == std::streampos(3000000000LL));
  return 0;
}
