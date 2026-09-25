// Clamping the witness index instead of assuming its range must not weaken the
// check Phase 2C exists for: `b` is not in the assigns clause, so writing
// through it is still a violation (#6513).
typedef struct
{
  int len;
  int data[4];
} buf_t;

int g;

void f(buf_t *b)
{
  __ESBMC_requires(b != 0);
  __ESBMC_assigns(g);
  __ESBMC_ensures(1);
  g = 1;
  b->data[2] = 99;
}

int main()
{
  return 0;
}
