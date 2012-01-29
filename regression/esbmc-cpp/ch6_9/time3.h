// Fig. 6.18: time3.h
// Declaration of class Time.
// Member functions defined in time3.cpp

// prevent multiple inclusions of header file
#ifndef TIME3_H  
#define TIME3_H 

class Time {

public:
   Time( int = 0, int = 0, int = 0 );  // default constructor

   // set functions
   void setTime( int, int, int );  // set hour, minute, second
   void setHour( int );   // set hour
   void setMinute( int ); // set minute
   void setSecond( int ); // set second

   // get functions
   int getHour();         // return hour
   int getMinute();       // return minute
   int getSecond();       // return second

   void printUniversal(); // output universal-time format
   void printStandard();  // output standard-time format

private:
   int hour;              // 0 - 23 (24-hour clock format)
   int minute;            // 0 - 59
   int second;            // 0 - 59

}; // end clas Time

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
