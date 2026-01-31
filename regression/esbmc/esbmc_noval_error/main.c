// Test: __ESBMC_EXTERN_NOVAL should error on non-extern variables
__ESBMC_EXTERN_NOVAL int counter = 42;

int main() {
    return counter;
}
