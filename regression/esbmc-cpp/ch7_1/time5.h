// Fig. 7.1: time5.h
// Definition of class Time.
// Member functions defined in time5.cpp.
#ifndef TIME5_H
#define TIME5_H

class Time {

public:
   Time( int = 0, int = 0, int = 0 );  // default constructor

   // set functions
   void setTime( int, int, int );  // set time
   void setHour( int );            // set hour
   void setMinute( int );          // set minute
   void setSecond( int );          // set second

   // get functions (normally declared const)
   int getHour() const;            // return hour
   int getMinute() const;          // return minute
   int getSecond() const;          // return second

   // print functions (normally declared const)
   void printUniversal() const;    // print universal time
   void printStandard();           // print standard time

private:
   int hour;    // 0 - 23 (24-hour clock format)
   int minute;  // 0 - 59
   int second;  // 0 - 59

}; // end class Time

#endif

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
