#include<stdio.h>
#include<stdlib.h>
#include<string.h>

unsigned int nondet_uint();
int nondet_int();

unsigned int N_LIN=nondet_uint();
unsigned int N_COL=nondet_uint();

main()
{
      
  unsigned int j,k;
  int matriz[N_COL][N_LIN], maior;
  
  maior = nondet_int();
  for(j=0;j<N_COL;j++)
    for(k=0;k<N_LIN;k++)
    {       
       matriz[j][k] = nondet_int();
       
       if(matriz[j][k]>maior)
          maior = matriz[j][k];                          
    }                       
    
  for(j=0;j<N_COL;j++)
    for(k=0;k<N_LIN;k++)
      assert(matriz[j][k]<=maior);    
}

