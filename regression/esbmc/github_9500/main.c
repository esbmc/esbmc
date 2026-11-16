#include <assert.h>
int nondet_int(void);

void rec(_Bool r, int *p) {
  int l = nondet_int();
  if (r) {
    *p = 20;
    __ESBMC_assume(l == 30);
  } else {
    rec(1, &l);
    assert(l == 20);
  }
}

int main(void) {
  rec(0, 0);
  return 0;
}
