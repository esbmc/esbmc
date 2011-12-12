// Fig. 10.24: employee.cpp
// Abstract-base-class Employee member-function definitions.
// Note: No definitions are given for pure virtual functions.
#include <iostream>

using std::cout;
using std::endl;

#include "employee.h"  // Employee class definition

// constructor
Employee::Employee( const string &first, const string &last,
   const string &SSN )
   : firstName( first ),
     lastName( last ),
     socialSecurityNumber( SSN )
{
   // empty body 

} // end Employee constructor

// return first name
string Employee::getFirstName() const 
{ 
   return firstName;  
   
} // end function getFirstName

// return last name
string Employee::getLastName() const
{
   return lastName;   
   
} // end function getLastName

// return social security number
string Employee::getSocialSecurityNumber() const
{
   return socialSecurityNumber;   
   
} // end function getSocialSecurityNumber

// set first name
void Employee::setFirstName( const string &first ) 
{ 
   firstName = first;  
   
} // end function setFirstName

// set last name
void Employee::setLastName( const string &last )
{
   lastName = last;   
   
} // end function setLastName

// set social security number
void Employee::setSocialSecurityNumber( const string &number )
{
   socialSecurityNumber = number;  // should validate
   
} // end function setSocialSecurityNumber

// print Employee's information
void Employee::print() const
{ 
   cout << getFirstName() << ' ' << getLastName() 
        << "\nsocial security number: " 
        << getSocialSecurityNumber() << endl; 

} // end function print

/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/