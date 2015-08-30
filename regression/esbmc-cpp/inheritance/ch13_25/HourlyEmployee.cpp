// Fig. 13.18: HourlyEmployee.cpp
// Definições de função membro da classe HourlyEmployee.
#include <iostream>
using std::cout;

#include "HourlyEmployee.h" // Definição da classe HourlyEmployee

// construtor
HourlyEmployee::HourlyEmployee( const string &first, const string &last, 
   const string &ssn, double hourlyWage, double hoursWorked )
   : Employee( first, last, ssn )
{
   setWage( hourlyWage ); // valida a remuneração por hora
   setHours( hoursWorked ); // valida as horas trabalhadas
} // fim do construtor HourlyEmployee

// configura a remuneração
void HourlyEmployee::setWage( double hourlyWage ) 
{ 
   wage = ( hourlyWage < 0.0 ? 0.0 : hourlyWage ); 
} // fim da função setWage

// retorna a remuneração
double HourlyEmployee::getWage() const
{
   return wage;
} // fim da função getWage

// configura as horas trabalhadas
void HourlyEmployee::setHours( double hoursWorked )
{ 
   hours = ( ( ( hoursWorked >= 0.0 ) && ( hoursWorked <= 168.0 ) ) ?
      hoursWorked : 0.0 );
} // fim da função setHours

// retorna as horas trabalhadas
double HourlyEmployee::getHours() const
{
   return hours;
} // fim da função getHours

// calcula os rendimentos;
// sobrescreve a função virtual pura earnings em Employee
double HourlyEmployee::earnings() const 
{ 
   if ( getHours() <= 40 ) // nenhuma hora extra
      return getWage() * getHours();
   else               
      return 40 * getWage() + ( ( getHours() - 40 ) * getWage() * 1.5 );
} // fim da função earnings

// imprime informações do HourlyEmployee
void HourlyEmployee::print() const
{
   cout << "hourly employee: ";
   Employee::print(); // reutilização de código
   cout << "\nhourly wage: " << getWage() << 
      "; hours worked: " << getHours();
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
