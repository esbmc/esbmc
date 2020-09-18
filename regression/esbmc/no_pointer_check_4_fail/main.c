// Check if comparation is nondet
int main(void) {
  int p; // unique stack addr
  int q; // unique stack addr;

  if(p > q) __ESBMC_assert(0,"p shouldn't be greater than q");
  return 0;
}