// Fig. 13.17: HourlyEmployee.h
// Definição da classe HourlyEmployee.
#ifndef HOURLY_H
#define HOURLY_H

#include "Employee.h" // Definição da classe Employee

class HourlyEmployee : public Employee 
{
public:
   HourlyEmployee( const string &, const string &, 
      const string &, double = 0.0, double = 0.0 );
   
   void setWage( double ); // configura o salário por hora
   double getWage() const; // retorna o salário por hora

   void setHours( double ); // configura as horas trabalhadas
   double getHours() const; // retorna as horas trabalhadas

   // palavra-chave virtual assinala intenção de sobrescrever 
   virtual double earnings() const; // calcula os rendimentos 
   virtual void print() const; // imprime o objeto HourlyEmployee
private:
   double wage; // salário por hora
   double hours; // horas trabalhadas durante a semana
}; // fim da classe HourlyEmployee

#endif // HOURLY_H


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
