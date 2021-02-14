// Fig. 13.17: HourlyEmployee.h
// Defini��o da classe HourlyEmployee.
#ifndef HOURLY_H
#define HOURLY_H

#include "Employee.h" // Defini��o da classe Employee

class HourlyEmployee : public Employee 
{
public:
   HourlyEmployee( const string &, const string &, 
      const string &, double = 0.0, double = 0.0 );
   
   void setWage( double ); // configura o sal�rio por hora
   double getWage() const; // retorna o sal�rio por hora

   void setHours( double ); // configura as horas trabalhadas
   double getHours() const; // retorna as horas trabalhadas

   // palavra-chave virtual assinala inten��o de sobrescrever 
   virtual double earnings() const; // calcula os rendimentos 
   virtual void print() const; // imprime o objeto HourlyEmployee
private:
   double wage; // sal�rio por hora
   double hours; // horas trabalhadas durante a semana
}; // fim da classe HourlyEmployee

#endif // HOURLY_H


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
