// Fig. 12.15: BasePlusCommissionEmployee.cpp
// Defini��es de fun��o membro da classe BasePlusCommissionEmployee.
#include <iostream>
using std::cout;

// Defini��o da classe BasePlusCommissionEmployee
#include "BasePlusCommissionEmployee.h"

// construtor
BasePlusCommissionEmployee::BasePlusCommissionEmployee( 
   const string &first, const string &last, const string &ssn, 
   double sales, double rate, double salary ) 
   // chama explicitamente o construtor da classe b�sica             
   : CommissionEmployee( first, last, ssn, sales, rate )
{
   setBaseSalary( salary ); // valida e armazena o sal�rio-base
} // fim do construtor BasePlusCommissionEmployee

// configura o sal�rio-base
void BasePlusCommissionEmployee::setBaseSalary( double salary )
{
   baseSalary = ( salary < 0.0 ) ? 0.0 : salary;
} // fim da fun��o setBaseSalary

// retorna o sal�rio-base
double BasePlusCommissionEmployee::getBaseSalary() const
{
   return baseSalary;
} // fim da fun��o getBaseSalary

// calcula os rendimentos
double BasePlusCommissionEmployee::earnings() const
{
   // pode acessar dados protected da classe b�sica   
   return baseSalary + ( commissionRate * grossSales );
} // fim da fun��o earnings

// imprime o objeto BasePlusCommissionEmployee
void BasePlusCommissionEmployee::print() const
{
   // pode acessar dados protected da classe b�sica                     
   cout << "base-salaried commission employee: " << firstName << ' '    
      << lastName << "\nsocial security number: " << socialSecurityNumber
      << "\ngross sales: " << grossSales                                
      << "\ncommission rate: " << commissionRate                        
      << "\nbase salary: " << baseSalary;                               
} // fim da fun��o print


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
