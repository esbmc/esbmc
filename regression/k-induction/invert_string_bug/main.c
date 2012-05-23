//#include <stdio.h>
//#include <string.h>
//#define MAX 10
int nondet_int();
char nondet_char();
unsigned int nondet_uint();
int MAX = nondet_uint();

int main() {
    char str1[MAX], str2[MAX];
    int cont, i, j;
    cont = 0;
    
    for (i=0; i<MAX; i++) {
        str1[i]=nondet_char();
    }
	str1[MAX-1]= '\0';

    j = 0;
    
    // Copia str1 inversa para str2
    for (i = MAX - 1; i >= 0; i--) {
        str2[j] = str1[0];
        j++;
    }
	//__ESBMC_assume(i<0);
	j = MAX-1;
    for (i=0; i<MAX; i++) {
      assert(str1[i] == str2[j]);
	  j--;
    }
    
}
