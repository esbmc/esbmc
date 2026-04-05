#include <stdlib.h>

typedef union {
} b;

typedef struct {
  b c;
  int a;
} d;

d *e;

int main() {
  void *f = malloc(sizeof(d));
  e = f;
  e->c;
}
