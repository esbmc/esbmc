unsigned int nondet_uint();
int nondet_int();

#define SIZE 1

int main()
{
  unsigned int j,k;
  int array[SIZE], menor;
  
  menor = nondet_int();

  for(j=0;j<SIZE;j++) {
       array[j] = nondet_int();
       
       if(array[j]<=menor)
          menor = array[j];                          
    }                       
    
    assert(array[0]>menor);    
}

