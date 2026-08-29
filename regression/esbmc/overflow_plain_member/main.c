int nondet_int(void);

struct
{
  int a;
} b = {3};

int main()
{
  // The same arithmetic on a non-bitfield member is checked; pins the boundary
  // rather than --overflow-check as a whole.
  int a = nondet_int();
  b.a += a;
}
