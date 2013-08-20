// Fig. 12.16: fig12_16.cpp
// Testando a classe BasePlusCommissionEmployee.
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
   // instancia o objeto BasePlusCommissionEmployee             
   BasePlusCommissionEmployee                                   
      employee( "Bob", "Lewis", "333-33-3333", 5000, .04, 300 );
   
   // configura a formatação de saída de ponto flutuante
   cout << fixed << setprecision( 2 );

   // obtém os dados de empregado comissionado
   cout << "Employee information obtained by get functions: \n" 
      << "\nFirst name is " << employee.getFirstName() 
      << "\nLast name is " << employee.getLastName() 
      << "\nSocial securiti number is " 
      << employee.getSocialSecurityNumber() 
      << "\nGross sales is " << employee.getGrossSales() 
      << "\nCommission rate is " << employee.getCommissionRate()
      << "\nBase salary is " << employee.getBaseSalary() << endl;

   employee.setBaseSalary( 1000 ); // configura o salário-base

   cout << "\nUpdated employee information output by print function: \n" 
      << endl;
   employee.print(); // exibe as novas informações do empregado

   cout << "\n\nEmployee's earnings: $" << employee.earnings() << endl;

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
