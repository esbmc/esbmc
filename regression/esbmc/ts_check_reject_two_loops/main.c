int nondet_int();

int main() {
  int x = 0;
  while (nondet_int())
    x++;
  for (;;)
    __ESBMC_assert(x >= 0, "x stays non-negative");
}
