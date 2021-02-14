// Fig. 12.26: fig12_26.cpp
// Ordem de exibi��o em que a classe b�sica e construtores e destrutores 
// da classe derivada s�o chamados.
#include <iostream>
using std::cout;
using std::endl;
using std::fixed;

#include <iomanip>
using std::setprecision;

// Defini��o da classe BasePlusCommissionEmployee
#include "BasePlusCommissionEmployee.h" 

int main()
{
   // configura a formata��o de sa�da de ponto flutuante
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
 * NOTA DE ISEN��O DE RESPONSABILIDADES: Os autores e o editor deste      *
 * livro empregaram seus melhores esfor�os na prepara��o do livro. Esses  *
 * esfor�os incluem o desenvolvimento, pesquisa e teste das teorias e     *
 * programas para determinar sua efic�cia. Os autores e o editor n�o      *
 * oferecem nenhum tipo de garantia, expl�cita ou implicitamente, com     *
 * refer�ncia a esses programas ou � documenta��o contida nesses livros.  *
 * Os autores e o editor n�o ser�o respons�veis por quaisquer danos,      *
 * acidentais ou conseq�entes, relacionados com ou provenientes do        *
 * fornecimento, desempenho ou utiliza��o desses programas.               *
 **************************************************************************/
