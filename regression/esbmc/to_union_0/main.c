union foo { int i; double d; };
union foo FOO;

int main() {
    int x = __VERIFIER_nondet_int();
    __VERIFIER_assume(x > 10);

    FOO = (union foo) x;
    __ESBMC_assert(FOO.i > 10, "int value should initialize union foo properly");
    return 0;
}