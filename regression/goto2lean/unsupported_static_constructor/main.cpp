struct S { S() { __ESBMC_assert(0, "ctor runs before main"); } };
S s;
int main() { return 0; }
