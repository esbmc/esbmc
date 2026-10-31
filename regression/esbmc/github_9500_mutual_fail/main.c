#include <assert.h>
int nondet_int(void);
void g(_Bool r, int *p);

void f(_Bool r, int *p) {
  int l = nondet_int();
  if (r) {
    *p = 20;
    __ESBMC_assume(l == 30);
  } else {
    g(1, &l);
    assert(l != 20);
  }
}

void g(_Bool r, int *p) { f(r, p); }

int main(void) {
  f(0, 0);
  return 0;
}
