// The closed form counts in unsigned arithmetic, which the check would flag.
int a[100];
int main() {
  int j = -5;
  for (int i = 0; i < 100; i++) { a[i] = j; j += 1; }
  __ESBMC_assert(j == 95, "");
}
