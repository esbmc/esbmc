// Test basic assigns clause functionality
// This test demonstrates that precise havoc only modifies variables in assigns clause

int global_x = 0;
int global_y = 0;
int global_z = 0;

void modify_x() {
    __ESBMC_assigns("global_x");
    __ESBMC_ensures(global_x == 42);
    
    global_x = 42;
}

int main() {
    global_x = 0;
    global_y = 100;
    global_z = 200;
    
    // Call function with contract
    modify_x();
    
    // After replace-call:
    // - global_x should be havoc'd (can be any value, but ensures says it's 42)
    // - global_y should NOT be havoc'd (should remain 100)
    // - global_z should NOT be havoc'd (should remain 200)
    
    __ESBMC_assert(global_y == 100, "global_y unchanged by modify_x");
    __ESBMC_assert(global_z == 200, "global_z unchanged by modify_x");
    __ESBMC_assert(global_x == 42, "global_x modified per ensures");
    
    return 0;
}
