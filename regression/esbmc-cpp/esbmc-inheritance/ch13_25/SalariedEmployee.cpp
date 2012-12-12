// Fig. 13.16: SalariedEmployee.cpp
// Definições de função membro da classe SalariedEmployee.
#include <iostream>
using std::cout;

#include "SalariedEmployee.h" // Definição da classe SalariedEmployee

// construtor
SalariedEmployee::SalariedEmployee( const string &first, 
   const string &last, const string &ssn, double salary )
   : Employee( first, last, ssn )
{ 
   setWeeklySalary( salary ); 
} // fim do construtor SalariedEmployee

// configura o salário
void SalariedEmployee::setWeeklySalary( double salary )
{ 
   weeklySalary = ( salary < 0.0 ) ? 0.0 : salary; 
} // fim da função setWeeklySalary

// retorna o salário
double SalariedEmployee::getWeeklySalary() const
{
   return weeklySalary;
} // fim da função getWeeklySalary

// calcula os rendimentos;
// sobrescreve a função virtual pura earnings em Employee
double SalariedEmployee::earnings() const 
{ 
   return getWeeklySalary(); 
} // fim da função earnings

// imprime informações de SalariedEmployee
void SalariedEmployee::print() const
{
   cout << "salaried employee: ";
   Employee::print(); // reutiliza função print da classe básica abstrata
   cout << "\nweekly salary: " << getWeeklySalary();
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
