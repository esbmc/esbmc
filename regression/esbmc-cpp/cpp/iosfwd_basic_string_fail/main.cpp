// libstdc++ names std::basic_string from <iosfwd> alone (via <bits/stringfwd.h>),
// so a translation unit that forward-declares only must parse. <string> is
// deliberately not included: including it would supply the declaration and the
// test would no longer pin <iosfwd>.
#include <iosfwd>
#include <cassert>

std::basic_string<char> *p = 0;

int main()
{
  assert(p != 0);
  return 0;
}
