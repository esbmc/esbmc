#include <assert.h>
#include <string.h>
int main()
{
  char d[8];
  strcpy(d, "hello");
  assert(d[0] == 'h' && d[4] == 'o' && d[5] == '\0');
  char f[8];
  strncpy(f, "world", 6);
  assert(f[0] == 'w' && f[4] == 'd');
  return 0;
}
