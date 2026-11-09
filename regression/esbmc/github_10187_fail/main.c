// #10187: unnamed int and double parameters under --interval-analysis
int nondet_int(void);
void h(int);
void k(double);
int main()
{
  int x = nondet_int();
  h(2);
  k(1.0);
  __ESBMC_assert(x != 5, "x may be 5");
  return 0;
}
