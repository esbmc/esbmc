// #2175: with --function, the harness is reachable only through the retargeted
// entry call; k-induction must still havoc what its loop writes through p.
struct range
{
  unsigned start, end;
};

static int next(struct range *r, unsigned *out)
{
  if (r->start >= r->end)
    return 0;
  *out = r->start++;
  return 1;
}

void harness(void)
{
  struct range it = {0, 10};
  struct range *p = &it;
  unsigned i;
  unsigned *q = &i;
  while (next(p, q))
    __ESBMC_assert(i != 5, "i reaches 5");
}

int main(void)
{
  return 0;
}
