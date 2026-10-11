#include <assert.h>
#include <strings.h>

int main(void)
{
  assert(strcasecmp("Hello", "hELLO") == 0);
  assert(strcasecmp("abc", "ABD") > 0);
  char a[2] = {'a', 'b'};
  return strncasecmp(a, "ABC", 3);
}
