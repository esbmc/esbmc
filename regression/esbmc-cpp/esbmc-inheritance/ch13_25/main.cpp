// Fig. 13.25: fig13_25.cpp
// Demonstrando downcasting e o RTTI
// NOTA: Para esse exemplo executar em Visual C++ .NET,
// você precisa ativar o RTTI (Run-Time Type Info) para o projeto.
#include <iostream>
using std::cout;
using std::endl;
using std::fixed;

#include <iomanip>
using std::setprecision;
  
#include <vector>
using std::vector;

#include <typeinfo>

// inclui definições de classes na hierarquia Employee
#include "Employee.h"
#include "SalariedEmployee.h" 
#include "HourlyEmployee.h"
#include "CommissionEmployee.h"  
#include "BasePlusCommissionEmployee.h" 

int main()
{
   // configura a formatação de saída de ponto flutuante
   cout << fixed << setprecision( 2 ); 
   
   // cria um vector a partir dos quatro ponteiros da classe básica
   vector < Employee * > employees( 4 );

   // inicializa vector com vários tipos de Employees
   employees[ 0 ] = new SalariedEmployee(            
      "John", "Smith", "111-11-1111", 800 );         
   employees[ 1 ] = new HourlyEmployee(              
      "Karen", "Price", "222-22-2222", 16.75, 40 );  
   employees[ 2 ] = new CommissionEmployee(          
      "Sue", "Jones", "333-33-3333", 10000, .06 );   
   employees[ 3 ] = new BasePlusCommissionEmployee(  
      "Bob", "Lewis", "444-44-4444", 5000, .04, 300 );

   // processa polimorficamente cada elemento no vector employees
   for ( size_t i = 0; i < employees.size(); i++ ) 
   {
      employees[ i ]->print(); // gera saída de informações do empregado
      cout << endl;

      // ponteiro downcast                           
      BasePlusCommissionEmployee *derivedPtr =       
         dynamic_cast < BasePlusCommissionEmployee * >
            ( employees[ i ] );                      

      // determina se o elemento aponta para o empregado comissionado com
      // salário-base
      if ( derivedPtr != 0 ) // 0 se não for um BasePlusCommissionEmployee
      {
         double oldBaseSalary = derivedPtr->getBaseSalary();
         cout << "old base salary: $" << oldBaseSalary << endl;
         derivedPtr->setBaseSalary( 1.10 * oldBaseSalary );
         cout << "new base salary with 10% increase is: $" 
            << derivedPtr->getBaseSalary() << endl;
      } // fim do if
      
      cout << "earned $" << employees[ i ]->earnings() << "\n\n";
   } // fim do for
 
   // libera objetos apontados pelos elementos do vector
   for ( size_t j = 0; j < employees.size(); j++ ) 
   {
      // gera saída do nome de classe                
      cout << "deleting object of "                  
         << typeid( *employees[ j ] ).name() << endl;

      delete employees[ j ];
   } // fim do for

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
