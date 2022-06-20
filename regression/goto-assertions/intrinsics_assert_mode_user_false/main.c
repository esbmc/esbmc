int main() {
    __ESBMC_disable_assert_mode(1);
    __ESBMC_enable_assert_mode(1);    
    __ESBMC_assert(0, "Fail");
}