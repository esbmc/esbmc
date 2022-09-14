int main() {
	int a,b,c; // nondet-value
	assert(a); // can't cache
	assert(a && b); // should change to 'b'
	assert(a || c); // should change to '1'
}
