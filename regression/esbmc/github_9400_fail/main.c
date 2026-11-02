/* #9400: %ls of L"ab" writes two characters. */
#include <assert.h>
#include <stdio.h>
#include <wchar.h>

int main(void)
{
  char b[64];
  int r = snprintf(b, sizeof b, "%ls", L"ab");
  assert(r != 2);
  return 0;
}
