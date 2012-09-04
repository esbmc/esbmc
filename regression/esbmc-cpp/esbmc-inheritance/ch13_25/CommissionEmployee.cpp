// Fig. 13.20: CommissionEmployee.cpp
// Definições de função membro da classe CommissionEmployee.
#include <iostream>
using std::cout;

#include "CommissionEmployee.h" // Definição da classe CommissionEmployee

// construtor
CommissionEmployee::CommissionEmployee( const string &first, 
   const string &last, const string &ssn, double sales, double rate )
   : Employee( first, last, ssn )  
{
   setGrossSales( sales );
   setCommissionRate( rate );
} // fim do construtor CommissionEmployee

// configura a taxa de comissão
void CommissionEmployee::setCommissionRate( double rate )
{ 
   commissionRate = ( ( rate > 0.0 && rate < 1.0 ) ? rate : 0.0 );
} // fim da função setCommissionRate

// retorna a taxa de comissão
double CommissionEmployee::getCommissionRate() const
{
    return commissionRate;
} // fim da função getCommissionRate

// configura a quantidade de vendas brutas
void CommissionEmployee::setGrossSales( double sales ) 
{ 
   grossSales = ( ( sales < 0.0 ) ? 0.0 : sales ); 
} // fim da função setGrossSales

// retorna a quantidade de vendas brutas
double CommissionEmployee::getGrossSales() const
{
    return grossSales;
} // fim da função getGrossSales

// calcula os rendimentos;
// sobrescreve a função virtual pura earnings em Employee
double CommissionEmployee::earnings() const
{ 
   return getCommissionRate() * getGrossSales(); 
} // fim da função earnings

// imprime informações do CommissionEmployee
void CommissionEmployee::print() const
{
   cout << "commission employee: ";
   Employee::print(); // reutilização de código
   cout << "\ngross sales: " << getGrossSales() 
      << "; commission rate: " << getCommissionRate();
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
