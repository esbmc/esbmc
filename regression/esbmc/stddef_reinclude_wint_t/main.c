// clang's <stddef.h> is re-includable: a caller defines a __need_* macro to
// pull in one type, which is how <wchar.h> obtains wint_t. ESBMC's shadowing
// <stddef.h> carried a one-shot include guard, so any translation unit that
// reached <stddef.h> first left wint_t undeclared for every later include.
#include <stddef.h>
#include <wchar.h>
#include <assert.h>

int main(void)
{
  wint_t w = WEOF;
  size_t s = sizeof(int);
  ptrdiff_t d = (char *)&s - (char *)&s;

  assert(w == WEOF);
  assert(s == sizeof(int));
  assert(d == 0);
  return 0;
}
