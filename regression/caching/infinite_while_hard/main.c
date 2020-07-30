int main()
{
    int a = 1;

    while(__VERIFIER_nondet_int()) {
        assert(a);
        a++;
        if(a == 0) break;
        else a = __VERIFIER_nondet_int() % 100 + 1;
    }

    return 0;
}