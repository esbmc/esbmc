/* Negative: the anonymous union overlay is really modelled, not nondet. */
struct Reg {
  union {
    struct { int lo, hi; };
    int words[2];
  };
};

int main(void)
{
  struct Reg r;
  r.lo = 1;
  __CPROVER_assert(r.words[0] == 2, "wrong overlay value");
  return 0;
}
