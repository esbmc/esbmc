// Negative counterpart of github_7548_posix: strsep advances *stringp past the
// delimiter, which a call with no model behind it would leave untouched.
#include <cassert>
#include <cstring>

int main()
{
  char buffer[] = "a,b";
  char *rest = buffer;
  strsep(&rest, ",");
  assert(rest == buffer);
  return 0;
}
