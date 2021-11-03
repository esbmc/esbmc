/* Contributed by Anton Vasilyev. */

#include <stdlib.h>
#include <string.h>

struct A
{
  unsigned char a;
  unsigned char b : 2;
  unsigned char c : 2;
  unsigned char pad1[2];
  unsigned int d;
  unsigned char e;
  unsigned char pad2[3];
} __attribute__((packed));

struct A d;
int main(void)
{
  struct A *p;
  p = malloc(12);
  d.a = 1;
  d.b = 2;
  d.c = 3;
  d.d = 4;
  d.e = 5;
  memcpy(p, &d, 4);
  if(p->a != 1)
  {
    free(p);
  }
  if(p->b != 2)
  {
    free(p);
  }
  if(p->c != 3)
  {
    free(p);
  }
  if(p->d != 4)
  { //ERROR: offsetof(struct A, d) == 4, wasn't copied
    free(p);
  }
  free(p);
}
