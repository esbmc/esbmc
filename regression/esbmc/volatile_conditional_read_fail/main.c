volatile int left;
volatile int right;
int nondet_int(void);

int main(void)
{
  int selector = nondet_int();
  int first = *(selector ? &left : &right);
  int second = *(selector ? &left : &right);
  __ESBMC_assert(first == second, "volatile conditional reads may differ");
}
