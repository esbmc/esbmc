// Fig. 12.16: fig12_16.cpp
// Testando a classe BasePlusCommissionEmployee.
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
   // instancia o objeto BasePlusCommissionEmployee             
   BasePlusCommissionEmployee                                   
      employee( "Bob", "Lewis", "333-33-3333", 5000, .04, 300 );
   
   // configura a formata��o de sa�da de ponto flutuante
   cout << fixed << setprecision( 2 );

   // obt�m os dados de empregado comissionado
   cout << "Employee information obtained by get functions: \n" 
      << "\nFirst name is " << employee.getFirstName() 
      << "\nLast name is " << employee.getLastName() 
      << "\nSocial securiti number is " 
      << employee.getSocialSecurityNumber() 
      << "\nGross sales is " << employee.getGrossSales() 
      << "\nCommission rate is " << employee.getCommissionRate()
      << "\nBase salary is " << employee.getBaseSalary() << endl;

   employee.setBaseSalary( 1000 ); // configura o sal�rio-base

   cout << "\nUpdated employee information output by print function: \n" 
      << endl;
   employee.print(); // exibe as novas informa��es do empregado

   cout << "\n\nEmployee's earnings: $" << employee.earnings() << endl;

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
