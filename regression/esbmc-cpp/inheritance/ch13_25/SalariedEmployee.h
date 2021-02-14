// Fig. 13.15: SalariedEmployee.h
// Classe SalariedEmployee derivada de Employee.
#ifndef SALARIED_H
#define SALARIED_H

#include "Employee.h" // Defini��o da classe Employee

class SalariedEmployee : public Employee
{
public:
   SalariedEmployee( const string &, const string &, 
      const string &, double = 0.0 );

   void setWeeklySalary( double ); // configura o sal�rio semanal
   double getWeeklySalary() const; // retorna o sal�rio semanal

   // palavra-chave virtual assinala inten��o de sobrescrever   
   virtual double earnings() const; // calcula os rendimentos   
   virtual void print() const; // imprime objeto SalariedEmployee
private:
   double weeklySalary; // sal�rio por semana
}; // fim da classe SalariedEmployee

#endif // SALARIED_H


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
