// Fig. 13.13: Employee.h
// Classe b�sica abstrata Employee.
#ifndef EMPLOYEE_H
#define EMPLOYEE_H

#include <string> // classe string padr�o C++ 
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

   // a fun��o virtual pura cria a classe b�sica abstrata Employee
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
