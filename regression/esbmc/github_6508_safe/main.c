#include <stdlib.h>

struct S
{
  int buf[4];
  unsigned n;
};

/* The struct-hack idiom: `qux' is over-allocated past its declared bound, so
   the declared bound must not be enforced (see 82b5ce54f7). The char variant
   also carries trailing padding, which must not make `qux' look interior. */
struct H
{
  int n;
  int qux[1];
};

struct HC
{
  int n;
  char qux[1];
};

/* The exemption has to survive a nested member chain, not just a direct one. */
struct W
{
  long tag;
  struct H h;
};

/* A union's storage is deliberately shared between its members. `long long'
   rather than `long' so the union is 8 bytes under LLP64 (Windows) too and the
   write below stays inside it. */
union U
{
  char c[4];
  long long l;
};

int main()
{
  struct S s;
  struct S *p = &s;
  p->n = 3;
  p->buf[p->n] = 1;

  struct H *h = malloc(sizeof(struct H) + 3 * sizeof(int));
  if (h)
  {
    for (int i = 0; i < 4; i++)
      h->qux[i] = i;
    free(h);
  }

  struct HC *hc = malloc(sizeof(struct HC) + 16);
  if (hc)
  {
    for (int i = 0; i < 8; i++)
      hc->qux[i] = (char)i;
    free(hc);
  }

  struct W *w = malloc(sizeof(struct W) + 3 * sizeof(int));
  if (w)
  {
    w->h.qux[3] = 1;
    free(w);
  }

  union U u;
  union U *up = &u;
  up->l = 0;
  up->c[6] = 1;

  return 0;
}
