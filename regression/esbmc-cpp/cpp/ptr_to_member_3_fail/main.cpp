struct S
{
    int x;

    int getX()
    {
        return x;
    }
};

int main()
{
    S s{42};

    int (S::*pmf)() = &S::getX;

    __ESBMC_assert((s.*pmf)() != 42, "");
}