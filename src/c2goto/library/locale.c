#include <stddef.h>
#include <stdbool.h>
#include <locale.h>
#include <string.h>

/* Whether LC_COLLATE is the C locale, the only collation modelled. */
_Bool __ESBMC_c_collation = 1;

char *setlocale(int category, const char *locale)
{
__ESBMC_HIDE:;
  // If locale is NULL, just query current locale
  if (locale == NULL)
  {
    static char current_locale[32] = "C";
    return current_locale;
  }

  // Non-deterministically model success or failure
  _Bool success = nondet_bool();

  if (!success)
    return NULL;

  if (category == LC_COLLATE || category == LC_ALL)
    __ESBMC_c_collation =
      strcmp(locale, "C") == 0 || strcmp(locale, "POSIX") == 0;

  // On success, return a locale string
  static char result_locale[32];
  size_t len;
  __ESBMC_assume(len > 0 && len < 32);
  result_locale[len] = '\0';

  return result_locale;
}