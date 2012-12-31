// Fig. 13.13: Employee.h
// Classe básica abstrata Employee.
#ifndef EMPLOYEE_H
#define EMPLOYEE_H

#include <string> // classe string padrão C++ 
using std::string;

class Employee 
{
public:
   Employee( const string &, const string &, const string & );

   void setFirstName( const string & ); // configura o nome
   string getFirstName() const; // retorna o nome

   void setLastName( const string & ); // configura o sobrenome
   string getLastName() const; // retorna o sobrenome

   void setSocialSecurityNumber( const string & ); // configura o SSN
   string getSocialSecurityNumber() const; // retorna o SSN 

   // a função virtual pura cria a classe básica abstrata Employee
   virtual double earnings() const = 0; // virtual pura          
   virtual void print() const; // virtual                        
private:
   string firstName;
   string lastName;
   string socialSecurityNumber;
}; // fim da classe Employee

#endif // EMPLOYEE_H


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
