// Fig. 10.30: commission.cpp
// CommissionEmployee class member-function definitions.
#include <iostream>

using std::cout;

#include "commission.h"  // Commission class

// CommissionEmployee constructor 
CommissionEmployee::CommissionEmployee( const string &first,
   const string &last, const string &socialSecurityNumber,
   double grossWeeklySales, double percent )
   : Employee( first, last, socialSecurityNumber )  
{
   setGrossSales( grossWeeklySales );
   setCommissionRate( percent );

} // end CommissionEmployee constructor

// return commission worker's rate
double CommissionEmployee::getCommissionRate() const
{
    return commissionRate;

} // end function getCommissionRate

// return commission worker's gross sales amount
double CommissionEmployee::getGrossSales() const
{
    return grossSales;

} // end function getGrossSales

// set commission worker's weekly base salary
void CommissionEmployee::setGrossSales( double sales ) 
{ 
   grossSales = sales < 0.0 ? 0.0 : sales; 

} // end function setGrossSales

// set commission worker's commission
void CommissionEmployee::setCommissionRate( double rate )
{ 
    commissionRate = ( rate > 0.0 && rate < 1.0 ) ? rate : 0.0;

} // end function setCommissionRate

// calculate commission worker's earnings
double CommissionEmployee::earnings() const
{ 
   return getCommissionRate() * getGrossSales(); 

} // end function earnings

// print commission worker's name 
void CommissionEmployee::print() const
{
   cout << "\ncommission employee: ";
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