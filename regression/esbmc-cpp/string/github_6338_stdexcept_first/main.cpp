// github #6338: pins the reverse include order. <stdexcept> needs std::string
// for its what_arg constructors and <string> needs std::out_of_range for at(),
// so the two headers are mutually dependent; the cycle must resolve whichever
// one the translation unit names first.
#include <stdexcept>
#include <string>
#include <cassert>

int main()
{
  // <stdexcept> alone must still bring in a usable std::string. Constructing a
  // what_arg-carrying exception is deliberately avoided: __refcnted_cstr
  // mallocs, and ESBMC models that as possibly failing, so it may throw
  // bad_alloc -- true before this change too, and orthogonal to include order.
  std::string msg = "boom";
  assert(msg.length() == 4);

  std::string s = "ab";
  bool caught = false;
  try
  {
    s.at(7);
  }
  catch (const std::out_of_range &)
  {
    caught = true;
  }
  assert(caught);
  return 0;
}
