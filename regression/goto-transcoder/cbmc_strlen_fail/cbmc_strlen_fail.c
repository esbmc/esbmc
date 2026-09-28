#include <assert.h>
#include <string.h>
int main()
{
  char a[6] = "hello";
  assert(strlen(a) == 4); // length is 5, not 4
  return 0;
}
