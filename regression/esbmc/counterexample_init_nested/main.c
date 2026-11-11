struct Inner
{
  int i;
  double d;
};

struct Outer
{
  struct Inner inner;
  int values[2];
};

int main(void)
{
  struct Outer obj;
  __ESBMC_init_object(&obj);
  __ESBMC_assume(
    obj.inner.i == 7 && obj.inner.d == 1.25 && obj.values[0] == 3 &&
    obj.values[1] == 4);
  __ESBMC_assert(
    !(obj.inner.i == 7 && obj.inner.d == 1.25 && obj.values[0] == 3 &&
      obj.values[1] == 4),
    "audit failure");
  return 0;
}
