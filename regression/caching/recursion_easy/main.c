void foo(int a, int b) {
    if(b == 0) return;
    assert(a);
    foo(--a, --b);    
}

int main() {
    int a = __VERIFIER_nondet_int();
    int b = __VERIFIER_nondet_int();

    __ESBMC_assume(a > 10 && a < b);
    __ESBMC_assume(b > 50);

    foo(a,b);
    return 0;
}