// Fig. 13.16: SalariedEmployee.cpp
// Defini��es de fun��o membro da classe SalariedEmployee.
#include <iostream>
using std::cout;

#include "SalariedEmployee.h" // Defini��o da classe SalariedEmployee

// construtor
SalariedEmployee::SalariedEmployee( const string &first, 
   const string &last, const string &ssn, double salary )
   : Employee( first, last, ssn )
{ 
   setWeeklySalary( salary ); 
} // fim do construtor SalariedEmployee

// configura o sal�rio
void SalariedEmployee::setWeeklySalary( double salary )
{ 
   weeklySalary = ( salary < 0.0 ) ? 0.0 : salary; 
} // fim da fun��o setWeeklySalary

// retorna o sal�rio
double SalariedEmployee::getWeeklySalary() const
{
   return weeklySalary;
} // fim da fun��o getWeeklySalary

// calcula os rendimentos;
// sobrescreve a fun��o virtual pura earnings em Employee
double SalariedEmployee::earnings() const 
{ 
   return getWeeklySalary(); 
} // fim da fun��o earnings

// imprime informa��es de SalariedEmployee
void SalariedEmployee::print() const
{
   cout << "salaried employee: ";
   Employee::print(); // reutiliza fun��o print da classe b�sica abstrata
   cout << "\nweekly salary: " << getWeeklySalary();
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
