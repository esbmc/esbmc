#include <cassert>
#include <stdexcept>
#include <string>
int main()
{
  std::string s = "ab";
  int caught = 0;
  try
  {
    s.substr(3);
  }
  catch (const std::out_of_range &)
  {
    caught = 1;
  }
  assert(caught == 1);
  return 0;
}
