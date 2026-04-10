struct S
{
    int x;
    int y;
};

int main()
{
    S s{42};

    int S::*pm = &S::x;

    S *r = &s;

    __ESBMC_assert(r->*pm == 42, "");
}
