// #7797: a moved-from syncbuf must give up the buffer it wrapped.
#include <syncstream>
#include <streambuf>
#include <cassert>
#include <utility>

struct sink : std::streambuf
{
};

int main()
{
  sink out;
  std::syncbuf sb(&out);
  std::syncbuf moved(std::move(sb));
  assert(sb.get_wrapped() == &out);
  return 0;
}
