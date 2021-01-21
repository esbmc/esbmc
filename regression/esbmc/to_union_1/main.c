union foo { int i; double d; };
union foo FOO;

int main() {
    double x = __VERIFIER_nondet_double();
    __VERIFIER_assume(x > 10);

    FOO = (union foo) x;
    __ESBMC_assert(FOO.d > 10, "Initialized correctly");
    return 0;
}