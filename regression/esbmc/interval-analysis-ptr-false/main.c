int main()
{
 int a,*q;
 a = 1;
 q = &a;
 *q = 2;
 __ESBMC_assert(a != 2, "");
}
