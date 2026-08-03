#include <stdlib.h>
#include <assert.h>
int main(){
  size_t len = 2;
  char *p = malloc(len - 4);
  if (p) p[0] = 1;
  assert(0);
}
