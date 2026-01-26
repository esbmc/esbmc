struct S {
    int x;
    int y;
};


int main() {
    S s{42};

    int S::* pm = &S::x;

    __ESBMC_assert(s.*pm == 42, "");
}
