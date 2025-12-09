#include <assert.h>
#include <stdlib.h>
char *a = new char;
int main() {
  0 == realloc(a, 0);
  a[0] = 'a';
  a[1] = 'f';
  assert(a[0] != 'a');
}
