int main() {
        int foo;
        foo = 42;
        int asd = foo + 10;
        foo = 15;
        __ESBMC_assert(0, "bar");
        return 0;
}
