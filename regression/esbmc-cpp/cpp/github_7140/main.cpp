#include <cassert>
#include <string>

int main()
{
  std::string s("abc");

  // [string.require]: size() <= max_size() holds after every operation, and
  // capacity() cannot exceed max_size() either.
  assert(s.max_size() >= s.size());
  assert(s.max_size() >= s.capacity());

  std::string e;
  assert(e.max_size() > 0);

  return 0;
}
