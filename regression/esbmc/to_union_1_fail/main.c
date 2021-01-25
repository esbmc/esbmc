union foo { int i; double d; };
union foo FOO;

int main() {
    FOO = (union foo) FLT_MAX;
    __ESBMC_assert(0, "this assertion shouldn't be reached");
    return 0;
}