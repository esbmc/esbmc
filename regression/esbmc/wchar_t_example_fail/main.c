#include <assert.h>
#include <wchar.h>

int main()
{
  // Define a wide character string with at least 32 bytes
  wchar_t *str = L"Hello, this is a test string for wmemchr";

  // Character to search for
  wchar_t ch = L't';

  // Use wmemchr to find the first occurrence of the character
  wchar_t *result = wmemchr(str, ch, 32);

  assert(result != NULL);
  assert(result == str + 13); // should bei 12

  return 0;
}