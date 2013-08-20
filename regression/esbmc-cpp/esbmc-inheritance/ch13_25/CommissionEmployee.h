// Fig. 13.19: CommissionEmployee.h
// Classe CommissionEmployee derivada de Employee.
#ifndef COMMISSION_H
#define COMMISSION_H

#include "Employee.h" // Definição da classe Employee

class CommissionEmployee : public Employee 
{
public:
   CommissionEmployee( const string &, const string &,
      const string &, double = 0.0, double = 0.0 );

   void setCommissionRate( double ); // configura a taxa de comissão
   double getCommissionRate() const; // retorna a taxa de comissão

   void setGrossSales( double ); // configura a quantidade de vendas brutas
   double getGrossSales() const; // retorna a quantidade de vendas brutas

   // palavra-chave virtual assinala intenção de sobrescrever     
   virtual double earnings() const; // calcula os rendimentos     
   virtual void print() const; // imprime o objeto CommissionEmployee
private:
   double grossSales; // vendas brutas semanais
   double commissionRate; // porcentagem da comissão
}; // fim da classe CommissionEmployee

#endif // COMMISSION_H


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
