// Fig. 13.21: BasePlusCommissionEmployee.h
// Classe BasePlusCommissionEmployee derivada de Employee.
#ifndef BASEPLUS_H
#define BASEPLUS_H

#include "CommissionEmployee.h" // Defini��o da classe CommissionEmployee

class BasePlusCommissionEmployee : public CommissionEmployee 
{
public:
   BasePlusCommissionEmployee( const string &, const string &,
      const string &, double = 0.0, double = 0.0, double = 0.0 );

   void setBaseSalary( double ); // configura o sal�rio-base
   double getBaseSalary() const; // retorna o sal�rio-base
   
   // palavra-chave virtual assinala inten��o de sobrescrever               
   virtual double earnings() const; // calcula os rendimentos               
   virtual void print() const; // imprime o objeto BasePlusCommissionEmployee
private:
   double baseSalary; // sal�rio-base por semana
}; // fim da classe BasePlusCommissionEmployee

#endif // BASEPLUS_H


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
