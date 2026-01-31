// Test __ESBMC_EXTERN_NOVAL prevents nondet assignment for extern variables
__ESBMC_EXTERN_NOVAL extern int counter;

// Definition (simulates separate translation unit)
int counter = 42;

int main() {
    __ESBMC_assert(counter == 42, "Counter should have defined value");
    return 0;
}
