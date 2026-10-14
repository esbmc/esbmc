// #7797: <mdspan> maps multidimensional indices onto one buffer.
#include <mdspan>
#include <cassert>
#include <cstddef>

int main()
{
  int buf[6] = {0, 1, 2, 3, 4, 5};

  std::mdspan<int, std::extents<size_t, 2, 3>> m(buf);
  assert(m.rank() == 2);
  assert(m.rank_dynamic() == 0);
  assert(m.static_extent(0) == 2 && m.static_extent(1) == 3);
  assert(m.extent(0) == 2 && m.extent(1) == 3);
  assert(m.size() == 6);
  assert(!m.empty());
  assert(m.data_handle() == buf);
  assert(m.is_exhaustive());

  // layout_right: the last index varies fastest.
  assert(m.stride(0) == 3 && m.stride(1) == 1);
  assert((m[0, 0]) == 0);
  assert((m[0, 2]) == 2);
  assert((m[1, 0]) == 3);
  assert((m[1, 2]) == 5);
  m[1, 1] = 42;
  assert(buf[4] == 42);

  // Dynamic extents.
  std::mdspan<int, std::dextents<size_t, 2>> d(buf, 3, 2);
  assert(d.rank() == 2 && d.rank_dynamic() == 2);
  assert(d.extent(0) == 3 && d.extent(1) == 2);
  assert(d.size() == 6);
  assert(d.stride(0) == 2 && d.stride(1) == 1);
  assert((d[2, 0]) == 42);

  // layout_left: the first index varies fastest.
  std::mdspan<int, std::extents<size_t, 2, 3>, std::layout_left> l(buf);
  assert(l.stride(0) == 1 && l.stride(1) == 2);
  assert((l[0, 0]) == 0);
  assert((l[1, 0]) == 1);
  assert((l[0, 1]) == 2);

  // Rank 1.
  std::mdspan<int, std::dextents<size_t, 1>> v(buf, 6);
  assert(v.rank() == 1 && v.extent(0) == 6 && v.size() == 6);
  assert((v[3]) == 3);

  std::extents<size_t, 2, 3> e1, e2;
  assert(e1 == e2);
  // Differing extents must compare unequal, or == could just return true.
  std::dextents<size_t, 2> u(2, 3), w(3, 2);
  assert(!(u == w));

  // required_span_size and the default-constructed forms.
  assert(m.mapping().required_span_size() == 6);
  assert(m.is_always_exhaustive());
  assert(m.accessor().offset(buf, 2) == buf + 2);
  return 0;
}
