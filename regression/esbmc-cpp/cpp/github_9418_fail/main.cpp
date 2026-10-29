#include <cassert>
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
  assert(!thrown);
  return 0;
}
