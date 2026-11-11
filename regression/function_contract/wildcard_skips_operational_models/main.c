#include <string.h>
int f(int x) { __ESBMC_requires(x > 0); __ESBMC_ensures(__ESBMC_return_value > 0); return x; }
int main() { char s[3] = "ab"; __ESBMC_assert(strlen(s) == 2, "l"); return f(1); }
