// Fig. 12.23: CommissionEmployee.cpp
// Definições de função membro da classe CommissionEmployee.
#include <iostream>
using std::cout;
using std::endl;

#include "CommissionEmployee.h" // Definição da classe CommissionEmployee

// construtor
CommissionEmployee::CommissionEmployee( 
   const string &first, const string &last, const string &ssn, 
   double sales, double rate )
   : firstName( first ), lastName( last ), socialSecurityNumber( ssn )
{
   setGrossSales( sales ); // valida e armazena as vendas brutas
   setCommissionRate( rate ); // valida e armazena a taxa de comissão

   cout << "CommissionEmployee constructor: " << endl;
   print();
   cout << "\n\n";
} // fim do construtor CommissionEmployee

// destrutor                                        
CommissionEmployee::~CommissionEmployee()           
{                                                   
   cout << "CommissionEmployee destructor: " << endl;
   print();                                         
   cout << "\n\n";                                  
} // fim do destrutor CommissionEmployee            

// configura o nome
void CommissionEmployee::setFirstName( const string &first )
{
   firstName = first; // deve validar
} // fim da função setFirstName

// retorna o nome
string CommissionEmployee::getFirstName() const
{
   return firstName;
} // fim da função getFirstName

// configura o sobrenome
void CommissionEmployee::setLastName( const string &last )
{
   lastName = last; // deve validar
} // fim da função setLastName

// retorna o sobrenome
string CommissionEmployee::getLastName() const
{
   return lastName;
} // fim da função getLastName

// configura o SSN
void CommissionEmployee::setSocialSecurityNumber( const string &ssn )
{
   socialSecurityNumber = ssn; // deve validar
} // fim da função setSocialSecurityNumber

// retorna o SSN
string CommissionEmployee::getSocialSecurityNumber() const
{
   return socialSecurityNumber;
} // fim da função getSocialSecurityNumber

// configura a quantidade de vendas brutas
void CommissionEmployee::setGrossSales( double sales )
{
   grossSales = ( sales < 0.0 ) ? 0.0 : sales;
} // fim da função setGrossSales

// retorna a quantidade de vendas brutas
double CommissionEmployee::getGrossSales() const
{
   return grossSales;
} // fim da função getGrossSales

// configura a taxa de comissão
void CommissionEmployee::setCommissionRate( double rate )
{
   commissionRate = ( rate > 0.0 && rate < 1.0 ) ? rate : 0.0;
} // fim da função setCommissionRate

// retorna a taxa de comissão
double CommissionEmployee::getCommissionRate() const
{
   return commissionRate;
} // fim da função getCommissionRate

// calcula os rendimentos
double CommissionEmployee::earnings() const
{
   return getCommissionRate() * getGrossSales();
} // fim da função earnings

// imprime o objeto CommissionEmployee 
void CommissionEmployee::print() const
{
   cout << "commission employee: " 
      << getFirstName() << ' ' << getLastName() 
      << "\nsocial security number: " << getSocialSecurityNumber() 
      << "\ngross sales: " << getGrossSales() 
      << "\ncommission rate: " << getCommissionRate();
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
