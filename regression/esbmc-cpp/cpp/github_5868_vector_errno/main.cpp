#include <vector>
#include <cstdlib>
#include <cassert>

int main()
{
  // The shape nlohmann's lexer uses: errno around strtol, having included
  // only <vector> and <cstdlib>.
  errno = 0;
  const char *s = "12";
  char *end;
  long v = strtol(s, &end, 10);
  assert(v == 12);
  assert(errno == 0);
  return 0;
}
