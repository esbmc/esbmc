#include <cassert>
#include <string>

int main()
{
  std::string s = "abc";

  // Every character is visited exactly once.
  int n = 0, count = 0;
  for (std::string::iterator i = s.begin(); i != s.end(); ++i)
  {
    n += *i;
    count++;
  }
  assert(count == 3);
  assert(n == 'a' + 'b' + 'c');

  // Range-for is the same traversal.
  int m = 0;
  for (char c : s)
    m += c;
  assert(m == n);

  // An empty string iterates zero times.
  std::string e;
  int k = 0;
  for (std::string::iterator i = e.begin(); i != e.end(); ++i)
    k++;
  assert(k == 0);
  assert(e.begin() == e.end());

  // A one-character string iterates once.
  std::string one = "z";
  int j = 0;
  for (std::string::iterator i = one.begin(); i != one.end(); ++i)
    j++;
  assert(j == 1);

  return 0;
}
