int main()
{
    int a;
    __ESBMC_assume(a > 0);
    while(__VERIFIER_nondet_int()) {
        assert(a);
        a--;
        if(a == 0) break;
    }

    return 0;
}