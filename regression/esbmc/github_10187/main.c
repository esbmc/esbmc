// #10187: unnamed int and double parameters under --interval-analysis
void h(int);
void k(double);
int main()
{
  int x = 3;
  h(2);
  k(1.0);
  __ESBMC_assert(x == 3, "x unchanged");
  return 0;
}
