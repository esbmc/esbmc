int g;
__ESBMC_logic int f(int x) { return x + g; }
int main() { return f(1); }
