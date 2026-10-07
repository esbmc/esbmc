/* A --function harness skips __CPROVER__start, and with it CBMC's
 * __CPROVER_initialize; the globals must still start from their values. */
typedef int v4 __attribute__((vector_size(16)));
struct inner
{
  int a[4];
  char *p;
};
struct outer
{
  struct inner in;
  long n;
};

static const unsigned long size = 16;
int counter;
struct outer zeroed;
struct outer set = {{{1, 2, 3, 4}, 0}, 7};
v4 zeroed_vec;
v4 set_vec = {1, 2, 3, 4};

void harness(void)
{
  __CPROVER_assert(size == 16, "valued global");
  __CPROVER_assert(counter == 0, "zero-initialised scalar");
  __CPROVER_assert(
    zeroed.in.a[3] == 0 && zeroed.in.p == 0 && zeroed.n == 0,
    "zero-initialised nested struct");
  __CPROVER_assert(set.in.a[2] == 3 && set.n == 7, "valued nested struct");
  __CPROVER_assert(zeroed_vec[1] == 0 && set_vec[3] == 4, "vector globals");
}

int main(void)
{
  return 0;
}
