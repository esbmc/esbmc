unsigned x0, x1, x2, n;
void step() { x2 = x1; x1 = x0; x0 = (n == 4); n++; __ESBMC_assert(x2 == 0, "x2 == 0"); }
int main() { while (1) step(); }
