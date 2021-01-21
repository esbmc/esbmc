union foo { int i; double d; };
union foo FOO;

int main() {
    char c = __VERIFIER_nondet_char();
    __VERIFIER_assume(c > 10);

    FOO = (union foo) c;
    __ESBMC_assert(FOO.i > 10, "Initialized correctly");
    return 0;
}