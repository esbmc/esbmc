union foo { int i; double d; };
union foo FOO;

int bar(union foo f) {
    return f.i;
}

int main() {
    char x = __VERIFIER_nondet_int();
    __VERIFIER_assume(x > 10);

    int v = bar((union foo) x);
    __ESBMC_assert(v > 10, "Initialized correctly");
    return 0;
}