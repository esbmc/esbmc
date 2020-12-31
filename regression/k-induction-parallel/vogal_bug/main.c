//  programa: C) NUMERO DE VOGAIS

#include<string.h>

#define MAX 10

char nondet_char();

main(void)
{
  char string_entrada[MAX], vetor_vogais[]={'a','A','e','E','i','I','o','O','u','U','\0'};; 
  unsigned int  i,j,cont, tam_string, n_caracter;

  for(i=0;i<MAX;i++)
    string_entrada[i] = nondet_char();
  __ESBMC_assume(string_entrada[MAX-1]=='\0');
  
  // captura o número de caracteres de da string
  n_caracter = 0;
  while(string_entrada[n_caracter]!='\0')
    n_caracter++;

  //conta as vogais
  cont = 0;
  for(i=0;i<n_caracter;i++)
     for(j=0;j<8;j++)
        if(string_entrada[i] == vetor_vogais[j])
           cont++;
           
  i=0;
  int cont_aux = 0;
  while(string_entrada[i]!='\0')
  {
    for(j=0;j<10;j++)
    {
        if(string_entrada[i] == vetor_vogais[j])
           cont_aux++;
    }       
    i++;       
  }    
  assert(cont_aux==cont);                          
}
