int main() {
    __ESBMC_set_assert_mode(0);        
    __ESBMC_set_assert_mode(1);    
    __ESBMC_assert(0, "Fail");
}