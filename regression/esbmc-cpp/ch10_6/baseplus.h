// Fig. 10.31: baseplus.h
// BasePlusCommissionEmployee class derived from Employee.
#ifndef BASEPLUS_H
#define BASEPLUS_H

#include "commission.h"  // Employee class definition

class BasePlusCommissionEmployee : public CommissionEmployee {

public:
   BasePlusCommissionEmployee( const string &, const string &,
      const string &, double = 0.0, double = 0.0, double = 0.0 );

   void setBaseSalary( double );
   double getBaseSalary() const;
   
   virtual double earnings() const;
   virtual void print() const;

private:
   double baseSalary;       // base salary per week

}; // end class BasePlusCommissionEmployee

#endif // BASEPLUS_H

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