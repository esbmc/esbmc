#include <stdlib.h>
#include <string.h>
#include <assert.h>

struct str
{
  unsigned char a : 3;
  int b : 31;
  unsigned char c : 7;
} var;

int main()
{
  var.a = 0;
  var.b = 78324;
  var.c = 1;
  assert(var.a == 0);
  assert(var.b == 78324);
  assert(var.c == 1);
  struct str *ptr;
  ptr = &var;
  assert(sizeof(struct str) == 12);
  printf("ptr->a = %d\n", ptr->a);
  printf("ptr->b = %d\n", ptr->b);
  printf("ptr->c = %d\n", ptr->c);
  assert(ptr->a == 0);
  assert(ptr->b == 78324);
  assert(ptr->c == 1);
  return 0;
}
