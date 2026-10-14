// The <cctype> functions reached through <iostream> must carry real semantics,
// not merely resolve: 'a' is not a digit (#7536).
#include <cassert>
#include <iostream>

int main()
{
  assert(isdigit('a'));
  return 0;
}
