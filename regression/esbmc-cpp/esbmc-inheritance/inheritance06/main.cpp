// Fig. 12.26: fig12_26.cpp
// Ordem de exibição em que a classe básica e construtores e destrutores 
// da classe derivada são chamados.
#include <iostream>
using std::cout;
using std::endl;
using std::fixed;

#include <iomanip>
using std::setprecision;

// Definição da classe BasePlusCommissionEmployee
#include "BasePlusCommissionEmployee.h" 

int main()
{
   // configura a formatação de saída de ponto flutuante
   cout << fixed << setprecision( 2 );

   { // inicia novo escopo                         
      CommissionEmployee employee1(                
         "Bob", "Lewis", "333-33-3333", 5000, .04 );
   } // termina o escopo                           

   cout << endl;
   BasePlusCommissionEmployee                                    
      employee2( "Lisa", "Jones", "555-55-5555", 2000, .06, 800 );
   
   cout << endl;
   BasePlusCommissionEmployee                                      
      employee3( "Mark", "Sands", "888-88-8888", 8000, .15, 2000 );
   cout << endl;
   return 0; 
} // fim de main


/**************************************************************************
 * (C) Copyright 1992-2005 Deitel & Associates, Inc. e                    *
 * Pearson Education, Inc. Todos os direitos reservados                   *
 *                                                                        *
 * NOTA DE ISENÇÃO DE RESPONSABILIDADES: Os autores e o editor deste      *
 * livro empregaram seus melhores esforços na preparação do livro. Esses  *
 * esforços incluem o desenvolvimento, pesquisa e teste das teorias e     *
 * programas para determinar sua eficácia. Os autores e o editor não      *
 * oferecem nenhum tipo de garantia, explícita ou implicitamente, com     *
 * referência a esses programas ou à documentação contida nesses livros.  *
 * Os autores e o editor não serão responsáveis por quaisquer danos,      *
 * acidentais ou conseqüentes, relacionados com ou provenientes do        *
 * fornecimento, desempenho ou utilização desses programas.               *
 **************************************************************************/
