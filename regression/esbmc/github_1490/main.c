int main()
{
	int k = __VERIFIER_nondet_int() % 1024;
	int b = __VERIFIER_nondet_int() != 0;
	int r = k & b;
	if (b) // this is line 6
		r |= 16;
	else
		r |= 64;
	assert(r < 10);
}
