// ios::exceptions() was declared and never defined, so it returned a
// nondeterministic iostate: after os.exceptions(failbit), both
// `exceptions() == failbit` and `exceptions() != failbit` FAILED.
//
// Note this models the mask as storage only. [ios.base] also has
// exceptions(mask) throw when the stream state already matches the mask; OM
// code cannot throw a catchable exception (see #6338), so that part is not
// modelled -- but the mask itself is now determinate rather than garbage.
#include <sstream>
#include <cassert>

int main()
{
  std::ostringstream os;
  // [ios.base.cons]: a new stream starts with an empty exception mask
  assert(os.exceptions() == std::ios_base::goodbit);
  os.exceptions(std::ios_base::failbit);
  assert(os.exceptions() == std::ios_base::failbit);
  os.exceptions(std::ios_base::badbit);
  assert(os.exceptions() == std::ios_base::badbit);
  return 0;
}
