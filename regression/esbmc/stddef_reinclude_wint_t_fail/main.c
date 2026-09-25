// Anti-vacuity twin of stddef_reinclude_wint_t: wint_t has to be a usable type
// with the target's width, not merely a name that parses.
#include <stddef.h>
#include <wchar.h>
#include <assert.h>

int main(void)
{
  wint_t w = WEOF;
  assert(w != WEOF);
  return 0;
}
