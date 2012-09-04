// Fig. 10.32: baseplus.cpp
// BasePlusCommissionEmployee member-function definitions.
#include <iostream>

using std::cout;

#include "baseplus.h"

// constructor for class BasePlusCommissionEmployee
BasePlusCommissionEmployee::BasePlusCommissionEmployee( 
   const string &first, const string &last, 
   const string &socialSecurityNumber, 
   double baseSalaryAmount,
   double grossSalesAmount, 
   double rate )
   : CommissionEmployee( first, last, socialSecurityNumber, 
     grossSalesAmount, rate )  
{
   setBaseSalary( baseSalaryAmount );

} // end BasePlusCommissionEmployee constructor

// set base-salaried commission worker's wage
void BasePlusCommissionEmployee::setBaseSalary( double salary )
{ 
   baseSalary = salary < 0.0 ? 0.0 : salary; 

} // end function setBaseSalary

// return base-salaried commission worker's base salary
double BasePlusCommissionEmployee::getBaseSalary() const
{ 
    return baseSalary; 

} // end function getBaseSalary

// return base-salaried commission worker's earnings
double BasePlusCommissionEmployee::earnings() const
{ 
    return getBaseSalary() + CommissionEmployee::earnings(); 

} // end function earnings

// print base-salaried commission worker's name 
void BasePlusCommissionEmployee::print() const
{
   cout << "\nbase-salaried commission worker: ";
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