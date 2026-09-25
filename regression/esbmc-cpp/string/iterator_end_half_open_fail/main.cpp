#include <cassert>
#include <string>

int main()
{
  std::string s = "abc";

  int count = 0;
  for (std::string::iterator i = s.begin(); i != s.end(); ++i)
    count++;

  // end() is one past the last character, so all three are visited.
  assert(count == 2);

  return 0;
}
