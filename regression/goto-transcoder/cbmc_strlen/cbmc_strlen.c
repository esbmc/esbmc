#include <assert.h>
#include <string.h>
int main()
{
  char a[6] = "hello";
  assert(strlen(a) == 5); // byte-loop body counts to the NUL
  char e[1] = "";
  assert(strlen(e) == 0);
  return 0;
}
