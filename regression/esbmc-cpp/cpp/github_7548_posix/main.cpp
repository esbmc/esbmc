// strsep and srandom were undeclared in C++ (github #7548).
#include <cassert>
#include <cstdlib>
#include <cstring>

int main()
{
  srandom(7);

  char buffer[] = "a,,b";
  char *rest = buffer;
  char *first = strsep(&rest, ",");
  assert(first == buffer && strcmp(first, "a") == 0);
  char *empty = strsep(&rest, ",");
  assert(empty == buffer + 2 && *empty == '\0');
  char *last = strsep(&rest, ",");
  assert(strcmp(last, "b") == 0 && rest == nullptr);
  assert(strsep(&rest, ",") == nullptr);
  return 0;
}
