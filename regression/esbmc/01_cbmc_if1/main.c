int nondet_int();
int main() {
	int i, j=nondet_int();

	i = 1;
	if(j > 0)
	  j += i ;
//		j += i ;
//	else
//		j = 0;

	assert(i != j);
}
