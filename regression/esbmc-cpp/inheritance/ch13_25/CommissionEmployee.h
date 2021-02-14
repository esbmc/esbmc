// Fig. 13.19: CommissionEmployee.h
// Classe CommissionEmployee derivada de Employee.
#ifndef COMMISSION_H
#define COMMISSION_H

#include "Employee.h" // Defini��o da classe Employee

class CommissionEmployee : public Employee 
{
public:
   CommissionEmployee( const string &, const string &,
      const string &, double = 0.0, double = 0.0 );

   void setCommissionRate( double ); // configura a taxa de comiss�o
   double getCommissionRate() const; // retorna a taxa de comiss�o

   void setGrossSales( double ); // configura a quantidade de vendas brutas
   double getGrossSales() const; // retorna a quantidade de vendas brutas

   // palavra-chave virtual assinala inten��o de sobrescrever     
   virtual double earnings() const; // calcula os rendimentos     
   virtual void print() const; // imprime o objeto CommissionEmployee
private:
   double grossSales; // vendas brutas semanais
   double commissionRate; // porcentagem da comiss�o
}; // fim da classe CommissionEmployee

#endif // COMMISSION_H


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
