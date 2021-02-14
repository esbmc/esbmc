// Fig. 13.14: Employee.cpp
// defini��es de fun��o membro da classe b�sica abstrata Employee.
// Nota: Nenhuma defini��o recebe fun��es virtuais puras.
#include <iostream>
using std::cout;

#include "Employee.h" // Defini��o da classe Employee

// construtor
Employee::Employee( const string &first, const string &last,
   const string &ssn )
   : firstName( first ), lastName( last ), socialSecurityNumber( ssn )
{
   // corpo vazio
} // fim do construtor Employee

// configura o nome
void Employee::setFirstName( const string &first ) 
{ 
   firstName = first;  
} // fim da fun��o setFirstName

// retorna o nome
string Employee::getFirstName() const 
{ 
   return firstName;  
} // fim da fun��o getFirstName

// configura o sobrenome
void Employee::setLastName( const string &last )
{
   lastName = last;   
} // fim da fun��o setLastName

// retorna o sobrenome
string Employee::getLastName() const
{
   return lastName;   
} // fim da fun��o getLastName

// configura o SSN
void Employee::setSocialSecurityNumber( const string &ssn )
{
   socialSecurityNumber = ssn; // deve validar
} // fim da fun��o setSocialSecurityNumber

// retorna o SSN
string Employee::getSocialSecurityNumber() const
{
   return socialSecurityNumber;   
} // fim da fun��o getSocialSecurityNumber

// imprime informa��es de Employee (virtual, mas n�o virtual pura)
void Employee::print() const
{ 
   cout << getFirstName() << ' ' << getLastName() 
      << "\nsocial security number: " << getSocialSecurityNumber(); 
} // fim da fun��o print


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
