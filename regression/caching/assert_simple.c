int main() {
    //unsigned n = __VERIFIER_nondet_int();
    //__VERIFIER_assume(n > 1);
    int c = 0;
    unsigned n = 3;
    int arr[n];

    for(int i = 0; i < n; i++) arr[i] = 0;
    for(int i = 0; i < n; i++) c++;
    if(arr[0] == 7) assert(0);
/*
    for(int i = 0; i < n; i++) {
        assert(arr[0] == 0); // This will hit the AND cache a lot!
        if(i == 1) arr[0] = 7;
    }
*/
    __ESBMC_assert(arr[c-1] == 0, ""); // This will create OR expressions
}