__ESBMC_logic int odd(int n);
__ESBMC_logic int even(int n) { return n == 0 ? 1 : odd(n - 1); }
__ESBMC_logic int odd(int n) { return n == 0 ? 0 : even(n - 1); }
int main() { return even(4); }
