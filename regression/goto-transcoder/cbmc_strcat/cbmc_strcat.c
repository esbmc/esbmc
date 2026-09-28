#include <assert.h>
#include <string.h>
int main()
{
  char e[8] = "ab";
  strcat(e, "cd");
  assert(e[2] == 'c' && e[3] == 'd' && e[4] == '\0');
  char g[8] = "xy";
  strncat(g, "zw", 2);
  assert(g[2] == 'z' && g[3] == 'w' && g[4] == '\0');
  return 0;
}
