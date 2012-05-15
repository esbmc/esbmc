char nondet_char();
unsigned int nondet_uint();


int main() {
    unsigned int max = nondet_uint();
    __ESBMC_assume(max>0 && max<2);
    char str1[max], str2[max];
    //unsigned int i, j;
    int i, j;

    for (i=0; i<max; i++) {
        str1[i]=nondet_char();
    }

    str1[max-1]= '\0';

    j = 0;
   
    // Copia str1 inversa para str2
    for (i = max - 1; i >= 0; i--) {
        str2[j] = str1[i];
        j++;
    }

    j = max-1;
    for (i=0; i<max; i++) {
      assert(str1[i] == str2[j]);
      j--;
    }   
}
