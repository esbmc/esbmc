#include <assert.h>
#include <string.h>
int main()
{
  char h[6] = "hello";
  assert(strrchr(h, 'l') == h + 2); // wrong: last 'l' is at index 3, not 2
  return 0;
}
