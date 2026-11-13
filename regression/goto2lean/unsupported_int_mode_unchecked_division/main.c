int nondet_int(void);
int main(void) { int x = nondet_int(), y = nondet_int(); int q = x / y; return 0; }
