int main() {
    __ESBMC_disable_assert_mode(8);
    __ESBMC_enable_assert_mode(8);
    int a,b;
    return a+b;
}