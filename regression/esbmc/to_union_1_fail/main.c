union foo { int i; double d; };
union foo FOO;

int main() {
    FOO = (union foo) FLT_MAX;
    __ESBMC_assert(FOO.d > 10, "Initialized correctly");
    return 0;
}