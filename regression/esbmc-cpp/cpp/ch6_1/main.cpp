// Fig. 6.1: fig06_01.cpp
// Create a structure, set its members, and print it.
#include <iostream>

using std::cout;
using std::endl;

#include <iomanip>

using std::setfill;
using std::setw;

// structure definition
struct Time {    
   int hour;     // 0-23 (24-hour clock format)
   int minute;   // 0-59
   int second;   // 0-59

}; // end struct Time

void printUniversal( const Time & );  // prototype
void printStandard( const Time & );   // prototype

int main()
{
   Time dinnerTime;         // variable of new type Time

   dinnerTime.hour = 18;    // set hour member of dinnerTime
   dinnerTime.minute = 30;  // set minute member of dinnerTime
   dinnerTime.second = 0;   // set second member of dinnerTime

   cout << "Dinner will be held at ";
   printUniversal( dinnerTime );
   cout << " universal time,\nwhich is ";
   printStandard( dinnerTime );
   cout << " standard time.\n";

   dinnerTime.hour = 10;    // set hour to invalid value
   dinnerTime.minute = 73;  // set minute to invalid value
   
   cout << "\nTime with invalid values: ";
   printUniversal( dinnerTime );
   cout << endl;

   return 0;  

} // end main

// print time in universal-time format
void printUniversal( const Time &t )
{
   
   cout << setfill( '0' ) << setw( 2 ) << t.hour << ":"
        << setw( 2 ) << t.minute << ":" 
        << setw( 2 ) << t.second;

} // end function printUniversal

// print time in standard-time format
void printStandard( const Time &t )
{
   cout << ( ( t.hour == 0 || t.hour == 12 ) ? 
             12 : t.hour % 12 ) << ":" << setfill( '0' )
        << setw( 2 ) << t.minute << ":" 
        << setw( 2 ) << t.second 
        << ( t.hour < 12 ? " AM" : " PM" );

/* 
   cout << "\n \n";

   cout << ( ( t.hour == 0 || t.hour == 12 ) ? 12 : t.hour % 12 ) 
	<< ":" << (t.minute < 10 ? "0":"") << t.minute << ":" 
        << (t.second < 10 ? "0":"") << t.second 
        << ( t.hour < 12 ? " da manha" : " da tarde" );
*/



} // end function printStandard

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
