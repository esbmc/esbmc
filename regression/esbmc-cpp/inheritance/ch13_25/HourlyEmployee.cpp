// Fig. 13.18: HourlyEmployee.cpp
// Defini��es de fun��o membro da classe HourlyEmployee.
#include <iostream>
using std::cout;

#include "HourlyEmployee.h" // Defini��o da classe HourlyEmployee

// construtor
HourlyEmployee::HourlyEmployee( const string &first, const string &last, 
   const string &ssn, double hourlyWage, double hoursWorked )
   : Employee( first, last, ssn )
{
   setWage( hourlyWage ); // valida a remunera��o por hora
   setHours( hoursWorked ); // valida as horas trabalhadas
} // fim do construtor HourlyEmployee

// configura a remunera��o
void HourlyEmployee::setWage( double hourlyWage ) 
{ 
   wage = ( hourlyWage < 0.0 ? 0.0 : hourlyWage ); 
} // fim da fun��o setWage

// retorna a remunera��o
double HourlyEmployee::getWage() const
{
   return wage;
} // fim da fun��o getWage

// configura as horas trabalhadas
void HourlyEmployee::setHours( double hoursWorked )
{ 
   hours = ( ( ( hoursWorked >= 0.0 ) && ( hoursWorked <= 168.0 ) ) ?
      hoursWorked : 0.0 );
} // fim da fun��o setHours

// retorna as horas trabalhadas
double HourlyEmployee::getHours() const
{
   return hours;
} // fim da fun��o getHours

// calcula os rendimentos;
// sobrescreve a fun��o virtual pura earnings em Employee
double HourlyEmployee::earnings() const 
{ 
   if ( getHours() <= 40 ) // nenhuma hora extra
      return getWage() * getHours();
   else               
      return 40 * getWage() + ( ( getHours() - 40 ) * getWage() * 1.5 );
} // fim da fun��o earnings

// imprime informa��es do HourlyEmployee
void HourlyEmployee::print() const
{
   cout << "hourly employee: ";
   Employee::print(); // reutiliza��o de c�digo
   cout << "\nhourly wage: " << getWage() << 
      "; hours worked: " << getHours();
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
