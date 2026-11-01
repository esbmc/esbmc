int h(int x) { return x; }
__ESBMC_logic int f(int x) { return h(x); }
int main() { return f(1); }
