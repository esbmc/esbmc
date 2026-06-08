void foo(char **q)
{
    q[5];
}

int main(int argc, char **argv)
{
    int a;
    __ESBMC_assume(a >= 5 && a <= 20);
    char *b[2][a + 1]; // notice that this is now a 2d array, was 1d before
    foo(b[0]);
    return 0;
}
