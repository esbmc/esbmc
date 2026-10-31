/* #9400: %ls of a character outside ASCII does not print its 4 wchar_t
   bytes: it gives -1 in the C locale and 2 in a UTF-8 one. */
#include <assert.h>
#include <stdio.h>
#include <wchar.h>

int main(void)
{
  char b[64];
  int r = snprintf(b, sizeof b, "%ls", L"\xe9");
  assert(r == 4);
  return 0;
}
