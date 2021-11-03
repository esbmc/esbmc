/* Contributed by Anton Vasilyev. */

#include <stdlib.h>
#include <string.h>

struct A
{
  unsigned char a;
  unsigned char b : 2;
  unsigned char c : 2;
  unsigned char d : 4;
  unsigned char e : 4;
  unsigned char pad;
  unsigned int f;
} __attribute__((packed));

struct B
{
  unsigned char a;
  unsigned char b : 2;
  unsigned char c : 3;
  unsigned char d : 4;
  unsigned char e;
  unsigned char f;
  unsigned char f1;
  unsigned char f2;
  unsigned char f3;
  unsigned char f4;
} __attribute__((packed));

struct A d = {.b = 3};
int main(void)
{
  struct B *p;
  p = malloc(sizeof(struct B));
  memcpy(p, &d, sizeof(struct B)); //ERROR: sizeof(struct B) > sizeof(struct A)
  if(p->a != 0)
  {
    free(p);
  }
  if(p->b != 3)
  {
    free(p);
  }
  if(p->c != 0)
  {
    free(p);
  }
  if(p->d != 0)
  {
    free(p);
  }
  if(p->e != 0)
  {
    free(p);
  }
  if(p->f != 0)
  {
    free(p);
  }
  if(p->f1 != 0)
  {
    free(p);
  }
  if(p->f2 != 0)
  {
    free(p);
  }
  if(p->f3 != 0)
  {
    free(p);
  }
  if(p->f4 != 0)
  {
    free(p);
  }
  free(p);
}
