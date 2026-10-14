/* C99 7.20.6.2: div, ldiv and lldiv were declared in stdlib.h but never
   defined, so both members of the returned struct were unconstrained
   (github #5868). */
#include <stdlib.h>
#include <assert.h>

int main()
{
  div_t d = div(7, 2);
  assert(d.quot == 3);
  assert(d.rem == 1);

  /* Truncation is toward zero, so a negative numerator gives a negative
     remainder and quot*denom + rem == numer still holds. */
  div_t n = div(-7, 2);
  assert(n.quot == -3);
  assert(n.rem == -1);
  assert(n.quot * 2 + n.rem == -7);

  div_t e = div(6, 3);
  assert(e.quot == 2);
  assert(e.rem == 0);

  ldiv_t l = ldiv(-9L, 4L);
  assert(l.quot == -2L);
  assert(l.rem == -1L);

  lldiv_t ll = lldiv(9LL, 4LL);
  assert(ll.quot == 2LL);
  assert(ll.rem == 1LL);
  return 0;
}
