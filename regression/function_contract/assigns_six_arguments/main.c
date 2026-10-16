int a, b, c, d, e, f;

__ESBMC_contract
void update(void)
{
    __ESBMC_assigns(a, b, c, d, e, f);

    a = 1;
    b = 2;
    c = 3;
    d = 4;
    e = 5;
    f = 6;
}

int main(void)
{
    update();
    return 0;
}