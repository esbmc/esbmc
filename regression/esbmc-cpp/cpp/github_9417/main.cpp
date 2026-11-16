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
  assert(thrown);

  thrown = false;
  std::string t("x");
  try
  {
    s.insert(4, t);
  }
  catch (std::out_of_range &)
  {
    thrown = true;
  }
  assert(thrown);

  s.insert(3, "d");
  assert(s == "abcd");
  s.insert(0, t);
  assert(s == "xabcd");
  return 0;
}
