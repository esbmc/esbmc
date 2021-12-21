#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
  const char src[9] = "testing!";
  char dest[9];
  strcpy(dest, "Heloooo!");
  __builtin_memcpy(dest, src, strlen(src) + 1);
  assert(dest[0] == 'e');
  return 0;
}
