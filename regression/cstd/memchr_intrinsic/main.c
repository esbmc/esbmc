#include <string.h>
#include <assert.h>

extern char nondet_char(void);

int main()
{
  char a[4] = {10, 20, 30, 40};
  char c = nondet_char();
  void *p = memchr(a, c, 4);

  /* If a match is reported, the byte at that address must equal c. */
  if (p != NULL)
    assert(*(unsigned char *)p == (unsigned char)c);

  /* If c is none of the stored bytes, the result must be NULL. */
  if (c != 10 && c != 20 && c != 30 && c != 40)
    assert(p == NULL);

  /* First match wins. */
  char b[6] = {9, 9, 7, 9, 7, 0};
  assert(memchr(b, 7, 6) == &b[2]);

  /* n == 0 returns NULL even when the byte is present. */
  assert(memchr(a, 10, 0) == NULL);

  /* n == 0 examines no bytes, so buf need not be valid (C11 7.24.5.1). */
  assert(memchr(NULL, 10, 0) == NULL);

  return 0;
}
