// Fig. 6.9: salesp.h
// SalesPerson class definition.
// Member functions defined in salesp.cpp.
#ifndef SALESP_H
#define SALESP_H

class SalesPerson {

public:
   SalesPerson();                // constructor
   void getSalesFromUser();      // input sales from keyboard
   void setSales( int, double ); // set sales for a month
   void printAnnualSales();      // summarize and print sales
 
private:
   double totalAnnualSales();	   // utility function
   double sales[ 12 ];           // 12 monthly sales figures

}; // end class SalesPerson

#endif

/**************************************************************************
 * (C) Copyright 1992-2002 by Deitel & Associates, Inc. and Prentice      *
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