int main() {
	int a,b,c; // nondet-value
        __ESBMC_assume(a != 0);
	assert(a); // can't cache
        __ESBMC_assume(b != 0);
	assert(a || b); // should change to 'b'
	assert(a || c); // should change to '1'
}
