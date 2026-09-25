#include <assert.h>
#include <string.h>
int main()
{
  char h[6] = "hello";
  assert(strrchr(h, 'l') == h + 3); // last match
  assert(strrchr(h, 'z') == 0);     // not found -> NULL
  return 0;
}
