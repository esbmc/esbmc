#include <stddef.h>
static size_t q, r;
#define RV ((char *)__ESBMC_return_value)
#define SZ(s) (__ESBMC_get_object_size(s) - __ESBMC_POINTER_OFFSET(s))
#define K(p, s) ((size_t)(__ESBMC_POINTER_OFFSET(p) - __ESBMC_POINTER_OFFSET(s)))
char *mychr(const char *s, int ch)
{
  __ESBMC_requires(__ESBMC_exists(&q, q < SZ(s) && s[q] == 0));
  __ESBMC_assigns();
  __ESBMC_ensures(
    RV == NULL ||
    (__ESBMC_same_object(RV, s) &&
     __ESBMC_POINTER_OFFSET(RV) >= __ESBMC_POINTER_OFFSET(s) &&
     K(RV, s) < SZ(s) &&
     s[K(RV, s)] == (char)ch &&
     __ESBMC_forall(&q, !(q < K(RV, s)) || (s[q] != (char)ch && s[q] != 0))));
  __ESBMC_ensures(
    RV != NULL ||
    __ESBMC_forall(&q, !(q < SZ(s) && s[q] == (char)ch) || __ESBMC_exists(&r, r < q && s[r] == 0)));
  while (*s && *s != (char)ch)
    s++;
  if (*s == (char)ch)
    return (char *)s;
  return NULL;
}
int main()
{
  char p[6]; for (int i = 0; i < 5; i++) p[i] = nondet_char(); p[5] = 0; p[0] = 'x';
  char *e = mychr(p, '.');
  __ESBMC_assert(e == NULL || (*e == '.' && e > p), "dot");
}
