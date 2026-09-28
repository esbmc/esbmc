#include <assert.h>
#include <string.h>
int main()
{
  char d[8];
  strcpy(d, "hello");
  assert(d[0] == 'x'); // copied 'h', not 'x'
  return 0;
}
