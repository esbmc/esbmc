int main() {
    __ESBMC_disable_assert_mode(2);

    __ESBMC_enable_assert_mode(2);
    int *a;
    *a;    
}