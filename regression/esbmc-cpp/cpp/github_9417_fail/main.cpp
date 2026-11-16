#include <cassert>
#include <stdexcept>
#include <string>

int main()
{
  std::string s("abc");
  bool thrown = false;
  try
  {
    s.insert(5, "x");
  }
  catch (std::out_of_range &)
  {
    thrown = true;
  }
  assert(!thrown);
  return 0;
}
