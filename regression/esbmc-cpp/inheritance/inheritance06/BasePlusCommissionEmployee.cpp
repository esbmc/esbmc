// Fig. 12.25: BasePlusCommissionEmployee.cpp
// Definições de funções membro da classe BasePlusCommissionEmployee.
#include <iostream>
using std::cout;
using std::endl;

// Definição da classe BasePlusCommissionEmployee
#include "BasePlusCommissionEmployee.h"

// construtor
BasePlusCommissionEmployee::BasePlusCommissionEmployee( 
   const string &first, const string &last, const string &ssn, 
   double sales, double rate, double salary ) 
   // chama explicitamente o construtor da classe básica 
   : CommissionEmployee( first, last, ssn, sales, rate )
{
   setBaseSalary( salary ); // valida e armazena salário-base

   cout << "BasePlusCommissionEmployee constructor: " << endl;
   print();
   cout << "\n\n";
} // fim do construtor BasePlusCommissionEmployee

// destrutor                                                
BasePlusCommissionEmployee::~BasePlusCommissionEmployee()   
{                                                           
   cout << "BasePlusCommissionEmployee destructor: " << endl;
   print();                                                 
   cout << "\n\n";                                          
} // fim do destrutor BasePlusCommissionEmployee            

// configura o salário-base
void BasePlusCommissionEmployee::setBaseSalary( double salary )
{
   baseSalary = ( salary < 0.0 ) ? 0.0 : salary;
} // fim da função setBaseSalary

// retorna o salário-base
double BasePlusCommissionEmployee::getBaseSalary() const
{
   return baseSalary;
} // fim da função getBaseSalary

// calcula os rendimentos
double BasePlusCommissionEmployee::earnings() const
{
   return getBaseSalary() + CommissionEmployee::earnings();
} // fim da função earnings

// imprime o objeto BasePlusCommissionEmployee
void BasePlusCommissionEmployee::print() const
{
   cout << "base-salaried ";

   // invoca a função print de CommissionEmployee
   CommissionEmployee::print(); 
   
   cout << "\nbase salary: " << getBaseSalary();
} // fim da função print


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
