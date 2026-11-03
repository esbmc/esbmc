// #8678: std::unordered_map::at on an absent key throws std::out_of_range and
// does not insert the key; a present key is read and written through the
// reference.
#include <cassert>
#include <unordered_map>
#include <stdexcept>

int main()
{
  std::unordered_map<int, int> m;
  m[1] = 10;
  m.at(1) = 11;
  assert(m.at(1) == 11);
  bool caught = false;
  try
  {
    m.at(2);
  }
  catch (const std::out_of_range &)
  {
    caught = true;
  }
  assert(caught);
  assert(m.size() == 1);
  return 0;
}
