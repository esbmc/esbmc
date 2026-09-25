// [array.overview] requires std::array to provide iterator and const_iterator
// member types. The model defined pointer/const_pointer and built
// reverse_iterator on them, but never named iterator/const_iterator, so generic
// code spelling std::array<T,N>::iterator failed with "no type named
// 'iterator'". begin()/end() already return pointer, so the typedefs just give
// those the names the standard requires.
//
// --unwind is needed only because an iterator loop over an OM container has no
// statically-visible bound; that is a separate, pre-existing property shared
// with std::vector and is not what this test is about.
#include <array>
#include <cassert>
#include <iterator>
int main()
{
  std::array<int, 3> a = {1, 2, 3};
  std::array<int, 3>::iterator it = a.begin();
  assert(*it == 1);
  ++it;
  assert(*it == 2);

  std::array<int, 3>::const_iterator ci = a.cbegin();
  assert(*ci == 1);

  const std::array<int, 3> &c = a;
  std::array<int, 3>::const_iterator ci2 = c.begin();
  assert(*ci2 == 1);

  // value_type via the iterator, the shape generic code uses
  std::iterator_traits<std::array<int, 3>::iterator>::value_type v = *a.begin();
  assert(v == 1);
  return 0;
}
