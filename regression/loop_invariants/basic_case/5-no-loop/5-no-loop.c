

 #include <assert.h>

 int main() {
     int i = 0;
     int sum = 0;
 
    __ESBMC_loop_invariant(i >= 0 && i <= 5000000 && sum == i * 10);    
    assert(sum == 50000000);     
    return 0;
 }