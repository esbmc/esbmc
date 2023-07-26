int main()
{
    double x = nondet_float();

	__ESBMC_assume(x >= 0);
	
	__ESBMC_assert(x == x, "");

    return 0;
}
