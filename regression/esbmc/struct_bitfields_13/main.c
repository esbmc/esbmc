/* Contributed by Anton Vasilyev. */

#include <stdlib.h>
#include <string.h>

struct A
{
  unsigned char a;
  unsigned char pad1[3];
  unsigned int b : 2;
  unsigned int c : 2;
  unsigned int d : 17;
  unsigned char e : 4;
  unsigned int f;
} __attribute__((packed));

struct B
{
  unsigned char a;
  unsigned char pad1[3];
  unsigned char b : 2;
  unsigned char c : 2;
  unsigned char d : 4;
  unsigned char e;
  unsigned char f;
  unsigned char f1;
  unsigned char f2;
  unsigned char f3;
  unsigned char f4;
} __attribute__((packed));

struct B d;
int main(void)
{
  struct A *p;
  p = malloc(sizeof(struct A));
  memcpy(p, &d, sizeof(struct A)); //ERROR: sizeof(struct A) > sizeof(struct B)
  if(p->a != 0)
  {
    free(p);
  }
  if(p->b != 0)
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
  free(p);
}
