#include <stdlib.h>
#include <string.h>
#include <assert.h>

struct str 
{
    unsigned char pad1:1;
    unsigned char pad2:1;
    unsigned int a:2;
    unsigned char pad3:5;
    unsigned int b:31;
} var;

int main() 
{
  var.a = 3;
  struct str *ptr;
  ptr = malloc(sizeof(struct str));
  *ptr = var;
  assert(ptr->a > 3);
  return 0;
}

