/* #9400: %ls converts each wide character, it does not copy wchar_t bytes. */
#include <assert.h>
#include <stdio.h>
#include <wchar.h>

int main(void)
{
  char b[64];
  int r = snprintf(b, sizeof b, "%ls", L"ab");
  assert(r == 2);
  assert(b[0] == 'a' && b[1] == 'b' && b[2] == '\0');
  r = snprintf(b, sizeof b, "%.1ls|%4ls", L"xy", L"z");
  assert(r == 6);
  assert(b[0] == 'x' && b[1] == '|' && b[2] == ' ' && b[5] == 'z');
  return 0;
}
