
int main() {
    int i = 0;
    int sum = 0;
    int secret = 42; 
    
    // Outer 50 times, middle 20 times, inner 50 times, total 50000 iterations
    while (i < 50)
    {   
        __ESBMC_loop_invariant(i >= 0 && i <= 50 && sum == i * 10000);
        int j = 0;
        while (j < 20)
        {
            __ESBMC_loop_invariant(j >= 0 && j <= 20);
            __ESBMC_loop_invariant(sum == i * 10000 + j * 500);
            __ESBMC_loop_invariant(i >= 0 && i <= 50);
    
            int k = 0;
            while (k < 50)
            {
                __ESBMC_loop_invariant(k >= 0 && k <= 50);
                __ESBMC_loop_invariant(sum == i * 10000 + j * 500 + k * 10);
                __ESBMC_loop_invariant(j >= 0 && j <= 20);
                __ESBMC_loop_invariant(i >= 0 && i <= 50);
                sum += 10;
                k++;
                secret = secret + 1;
            }
            j++;
        }
        i++;
    }
    
    assert(sum == 500000);     // 50 * 20 * 50 * 10 = 500000
    assert(secret == 50042);   // 42 + 50 * 20 * 50 = 50042
    return 0;
}
