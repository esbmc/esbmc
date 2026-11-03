#include <cassert>
#include <stdexcept>
#include <string_view>
int main()
{
  std::string_view s = "ab";
  int caught = 0;
  try
  {
    s.at(3);
  }
  catch (const std::out_of_range &)
  {
    caught = 1;
  }
  assert(caught == 1);
  return 0;
}
