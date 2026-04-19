// Fig. 10.33: fig10_33.cpp
// Driver for Employee hierarchy.
#include <iostream>
#include <cassert>
using std::cout;
using std::endl;
using std::fixed;

#include <iomanip>

using std::setprecision;
  
#include <vector>

using std::vector;

#include <typeinfo>

#include "employee.h"    // Employee base class 
#include "salaried.h"    // SalariedEmployee class 
#include "commission.h"  // CommissionEmployee class 
#include "baseplus.h"    // BasePlusCommissionEmployee class 
#include "hourly.h"      // HourlyEmployee class 


int main()
{

   // set floating-point output formatting
   cout << fixed << setprecision( 2 );
   // create vector employees
   vector < Employee * > employees( 4 );

   // initialize vector with Employees
   employees[ 0 ] = new SalariedEmployee( "John", "Smith", 
      "111-11-1111", 800.00 );
   employees[ 1 ] = new CommissionEmployee( "Sue", "Jones", 
      "222-22-2222", 10000, .06 );
   employees[ 2 ] = new BasePlusCommissionEmployee( "Bob", 
      "Lewis", "333-33-3333", 300, 5000, .04 );
   employees[ 3 ] = new HourlyEmployee( "Karen", "Price", 
      "444-44-4444", 16.75, 40 );

   // generically process each element in vector employees
   for ( int i = 0; i < employees.size() + 1; i++ ) {

      // output employee information
      employees[ i ]->print();  

      // downcast pointer
      BasePlusCommissionEmployee *commissionPtr = 
         dynamic_cast < BasePlusCommissionEmployee * >
            ( employees[ i ] );

      // determine whether element points to base-salaried 
      // commission worker
      if ( commissionPtr != 0 ) {
         cout << "old base salary: $" 
              << commissionPtr->getBaseSalary() << endl;
         commissionPtr->setBaseSalary( 
            1.10 * commissionPtr->getBaseSalary() );
         cout << "new base salary with 10% increase is: $" 
              << commissionPtr->getBaseSalary() << endl;

      } // end if
      assert(i>=0 && i<3);                              
      cout << "earned $" << employees[ i ]->earnings() << endl;

   } // end for   

   // release memory held by vector employees
   for ( int j = 0; j < employees.size(); j++ ) {

      // output class name
      cout << "\ndeleting object of " 
           << typeid( *employees[ j ] ).name();

      delete employees[ j ];

   } // end for

   cout << endl;

   return 0;

} // end main

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
