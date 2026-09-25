#include <stdlib.h>
#include <assert.h>
#include <limits.h>

int main()
{
  char *endptr;

  long long v1 = strtoll("12345", &endptr, 10);
  assert(v1 == 12345);
  assert(*endptr == '\0');
  assert(endptr != NULL);

  long long v2 = strtoll("-42", NULL, 10);
  assert(v2 == -42);

  long long v3 = strtoll("0x2A", NULL, 16);
  assert(v3 == 42);

  // LLONG_MAX + 1 should saturate rather than wrap or crash
  long long v4 = strtoll("9223372036854775808", NULL, 10);
  assert(v4 == LLONG_MAX);

  return 0;
}
