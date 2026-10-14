// timegm was undeclared in the bundled <ctime>, so any translation unit naming
// it stopped at "use of undeclared identifier" before verification began.
#include <ctime>
#include <cassert>

int main()
{
  struct tm t = {};
  t.tm_year = 102;
  t.tm_mon = 9;
  t.tm_mday = 9;
  t.tm_isdst = 1;

  timegm(&t);

  assert(t.tm_isdst == 0);
  return 0;
}
