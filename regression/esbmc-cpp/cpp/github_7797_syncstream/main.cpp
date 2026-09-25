// #7797: <syncstream> wraps a stream buffer, and a move hands that over.
#include <syncstream>
#include <streambuf>
#include <cassert>
#include <utility>

// basic_streambuf's constructor is protected, so a wrappable buffer has to be
// a derived one.
struct sink : std::streambuf
{
};

int main()
{
  sink out;

  std::syncbuf sb(&out);
  assert(sb.get_wrapped() == &out);
  // emit() reports whether there is a buffer to emit to.
  assert(sb.emit());

  std::syncbuf empty;
  assert(empty.get_wrapped() == nullptr);
  assert(!empty.emit());

  sb.set_emit_on_sync(true);

  // The move takes the wrapped buffer with it.
  std::syncbuf moved(std::move(sb));
  assert(moved.get_wrapped() == &out);
  assert(sb.get_wrapped() == nullptr);
  assert(!sb.emit());

  std::osyncstream os(&out);
  assert(os.get_wrapped() == &out);
  assert(os.rdbuf()->get_wrapped() == &out);
  os.emit();

  // The osyncstream(ostream&) overload is left out: it reads os.rdbuf(), and
  // ostream is `virtual public ios` over `ios : public ios_base`, the shape
  // regression/esbmc-cpp/inheritance/virtual_base_with_base pins as broken.

  return 0;
}
