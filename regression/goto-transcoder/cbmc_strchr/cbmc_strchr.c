#include <assert.h>
#include <string.h>
int main()
{
  char h[6] = "hello";
  assert(strchr(h, 'l') == h + 2); // first match
  assert(strchr(h, 'z') == 0);     // not found -> NULL
  return 0;
}
