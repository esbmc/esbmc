#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

int main()
{
  std::vector<int> v;
  bool thrown = false;
  try
  {
    v.reserve(v.max_size() + 1);
  }
  catch (std::length_error &)
  {
    thrown = true;
  }
  assert(thrown);

  thrown = false;
  try
  {
    v.reserve(static_cast<std::size_t>(-1));
  }
  catch (std::length_error &)
  {
    thrown = true;
  }
  assert(thrown);

  v.reserve(10);
  assert(v.capacity() >= 10 && v.empty());
  return 0;
}
