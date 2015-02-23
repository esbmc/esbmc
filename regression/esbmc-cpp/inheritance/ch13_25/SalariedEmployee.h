// Fig. 13.15: SalariedEmployee.h
// Classe SalariedEmployee derivada de Employee.
#ifndef SALARIED_H
#define SALARIED_H

#include "Employee.h" // Definição da classe Employee

class SalariedEmployee : public Employee
{
public:
   SalariedEmployee( const string &, const string &, 
      const string &, double = 0.0 );

   void setWeeklySalary( double ); // configura o salário semanal
   double getWeeklySalary() const; // retorna o salário semanal

   // palavra-chave virtual assinala intenção de sobrescrever   
   virtual double earnings() const; // calcula os rendimentos   
   virtual void print() const; // imprime objeto SalariedEmployee
private:
   double weeklySalary; // salário por semana
}; // fim da classe SalariedEmployee

#endif // SALARIED_H


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
