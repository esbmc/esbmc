unsigned nondet_uint(void);
unsigned f(unsigned x) { if (x) return 1; __builtin_unreachable(); }
int main(void) { unsigned r = f(nondet_uint()); return 0; }
