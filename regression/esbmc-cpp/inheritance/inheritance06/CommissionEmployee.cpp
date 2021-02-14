// Fig. 12.23: CommissionEmployee.cpp
// Defini��es de fun��o membro da classe CommissionEmployee.
#include <iostream>
using std::cout;
using std::endl;

#include "CommissionEmployee.h" // Defini��o da classe CommissionEmployee

// construtor
CommissionEmployee::CommissionEmployee( 
   const string &first, const string &last, const string &ssn, 
   double sales, double rate )
   : firstName( first ), lastName( last ), socialSecurityNumber( ssn )
{
   setGrossSales( sales ); // valida e armazena as vendas brutas
   setCommissionRate( rate ); // valida e armazena a taxa de comiss�o

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
} // fim da fun��o setFirstName

// retorna o nome
string CommissionEmployee::getFirstName() const
{
   return firstName;
} // fim da fun��o getFirstName

// configura o sobrenome
void CommissionEmployee::setLastName( const string &last )
{
   lastName = last; // deve validar
} // fim da fun��o setLastName

// retorna o sobrenome
string CommissionEmployee::getLastName() const
{
   return lastName;
} // fim da fun��o getLastName

// configura o SSN
void CommissionEmployee::setSocialSecurityNumber( const string &ssn )
{
   socialSecurityNumber = ssn; // deve validar
} // fim da fun��o setSocialSecurityNumber

// retorna o SSN
string CommissionEmployee::getSocialSecurityNumber() const
{
   return socialSecurityNumber;
} // fim da fun��o getSocialSecurityNumber

// configura a quantidade de vendas brutas
void CommissionEmployee::setGrossSales( double sales )
{
   grossSales = ( sales < 0.0 ) ? 0.0 : sales;
} // fim da fun��o setGrossSales

// retorna a quantidade de vendas brutas
double CommissionEmployee::getGrossSales() const
{
   return grossSales;
} // fim da fun��o getGrossSales

// configura a taxa de comiss�o
void CommissionEmployee::setCommissionRate( double rate )
{
   commissionRate = ( rate > 0.0 && rate < 1.0 ) ? rate : 0.0;
} // fim da fun��o setCommissionRate

// retorna a taxa de comiss�o
double CommissionEmployee::getCommissionRate() const
{
   return commissionRate;
} // fim da fun��o getCommissionRate

// calcula os rendimentos
double CommissionEmployee::earnings() const
{
   return getCommissionRate() * getGrossSales();
} // fim da fun��o earnings

// imprime o objeto CommissionEmployee 
void CommissionEmployee::print() const
{
   cout << "commission employee: " 
      << getFirstName() << ' ' << getLastName() 
      << "\nsocial security number: " << getSocialSecurityNumber() 
      << "\ngross sales: " << getGrossSales() 
      << "\ncommission rate: " << getCommissionRate();
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
