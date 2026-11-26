struct Inner
{
  int tag;
};

struct Cell
{
  int value;
  struct Inner inner;
};

int main(void)
{
  struct Cell obj[2][2];
  __ESBMC_init_object(&obj);
  __ESBMC_assume(
    obj[0][0].value == 1 && obj[0][0].inner.tag == 11 && obj[0][1].value == 2 &&
    obj[0][1].inner.tag == 12 && obj[1][0].value == 3 &&
    obj[1][0].inner.tag == 13 && obj[1][1].value == 4 &&
    obj[1][1].inner.tag == 14);
  __ESBMC_assert(
    (obj[0][0].value == 1 && obj[0][0].inner.tag == 11 &&
     obj[0][1].value == 2 && obj[0][1].inner.tag == 12 &&
     obj[1][0].value == 3 && obj[1][0].inner.tag == 13 &&
     obj[1][1].value == 4 && obj[1][1].inner.tag == 14),
    "aggregate value");
  return 0;
}
