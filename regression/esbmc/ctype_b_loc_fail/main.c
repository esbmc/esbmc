#include <assert.h>

extern const unsigned short int **__ctype_b_loc(void);

int main()
{
  /* '_' is punctuation (1 << 2), not alphanumeric (1 << 3). */
  assert((*__ctype_b_loc())['_'] & (1 << 3));
  return 0;
}
