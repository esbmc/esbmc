/* #9000: this->p in an assigns clause covers the field. */
struct C {
  int *p; int k;
  void set(int *q) { __ESBMC_assigns(this->p, this->k); p = q; k = 1; }
};
int g;
int main() { C c; c.p = 0; c.set(&g); return 0; }
