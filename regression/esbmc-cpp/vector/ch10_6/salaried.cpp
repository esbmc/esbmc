// Fig. 10.26: salaried.cpp
// SalariedEmployee class member-function definitions.
#include <iostream>

using std::cout;

#include "salaried.h" // SalariedEmployee class definition

// SalariedEmployee constructor 
SalariedEmployee::SalariedEmployee( const string &first, 
   const string &last, const string &socialSecurityNumber,
   double salary )
   : Employee( first, last, socialSecurityNumber )
{ 
   setWeeklySalary( salary ); 

} // end SalariedEmployee constructor

// set salaried worker's salary
void SalariedEmployee::setWeeklySalary( double salary )
{ 
   weeklySalary = salary < 0.0 ? 0.0 : salary; 

} // end function setWeeklySalary

// calculate salaried worker's pay
double SalariedEmployee::earnings() const 
{ 
   return getWeeklySalary(); 

} // end function earnings

// return salaried worker's salary
double SalariedEmployee::getWeeklySalary() const
{
   return weeklySalary;

} // end function getWeeklySalary

// print salaried worker's name 
void SalariedEmployee::print() const
{
   cout << "\nsalaried employee: ";
   Employee::print();  // code reuse

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