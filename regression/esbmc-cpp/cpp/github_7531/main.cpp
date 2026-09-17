// <iosfwd> forward-declared the concrete stream classes but none of the basic_*
// alias templates, so a TU including only <iosfwd> could not name
// std::basic_istream<char>. The aliases moved to <iosfwd>, where [iosfwd.syn]
// declares them; repeating them in <istream>/<ostream>/<fstream> would declare
// the default template argument twice, which clang rejects (#7531).
#include <iosfwd>

typedef std::basic_istream<char, std::char_traits<char> > IStream;
typedef std::basic_ostream<char, std::char_traits<char> > OStream;
typedef std::basic_iostream<char, std::char_traits<char> > IOStream;
typedef std::basic_ifstream<char, std::char_traits<char> > IFStream;
typedef std::basic_ofstream<char, std::char_traits<char> > OFStream;
typedef std::basic_fstream<char, std::char_traits<char> > FStream;

int main()
{
  IStream *a = 0;
  OStream *b = 0;
  IOStream *c = 0;
  IFStream *d = 0;
  OFStream *e = 0;
  FStream *f = 0;
  __ESBMC_assert(!a && !b && !c && !d && !e && !f, "iosfwd aliases usable");
  return 0;
}
