int main() {
	int a;
	__ESBMC_assume(a % 2 != 0);
	int obj1;
	char obj2;
	int *ptr = a % 2 == 0 ? &obj1 : &obj2; 
        memset(ptr, 0, 4); // We are definitely at &obj2
	return 0;
	
}
