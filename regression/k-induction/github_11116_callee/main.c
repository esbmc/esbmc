unsigned x0, x1, x2;
void step() { x2 = x1; x1 = x0; x0 = 0; __ESBMC_assert(x2 == 0, "x2 == 0"); }
int main() { while (1) step(); }
