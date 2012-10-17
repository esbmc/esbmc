// Fig. 12.14: BasePlusCommissionEmployee.h
// Classe BasePlusCommissionEmployee derivada da classe
// CommissionEmployee.
#ifndef BASEPLUS_H
#define BASEPLUS_H

#include <string> // classe string padrão C++ 
using std::string;

#include "CommissionEmployee.h" // Declaração da classe CommissionEmployee

class BasePlusCommissionEmployee : public CommissionEmployee
{
public:
   BasePlusCommissionEmployee( const string &, const string &, 
      const string &, double = 0.0, double = 0.0, double = 0.0 );
   
   void setBaseSalary( double ); // configura o salário-base
   double getBaseSalary() const; // retorna o salário-base

   double earnings() const; // calcula os rendimentos
   void print() const; // imprime o objeto BasePlusCommissionEmployee
private:
   double baseSalary; // salário-base
}; // fim da classe BasePlusCommissionEmployee

#endif


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
