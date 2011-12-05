// Fig. 16.19: savefile.cpp
// Program to enter user's contact information into a 
// server-side file.

#include <iostream>

using std::cerr;
using std::cout;
using std::cin;
using std::ios;

#include <fstream>

using std::ofstream;

#include <string>

using std::string;

#include <cstdlib>

int main()
{
   char postString[ 1024 ] = "";
   int contentLength = 0;
   
   // variables to store user data
   string dataString = "";
   string firstname = "";
   string lastname = "";
   string address = "";
   string town = "";
   string state = "";
   string zipcode = "";
   string country = "";
   string email = "";   
   
   // data was posted
   if ( getenv( "CONTENT_LENGTH" ) )
      contentLength = atoi( getenv( "CONTENT_LENGTH" ) );
   
   cin.read( postString, contentLength );
   dataString = postString;
   
   // search for first '+' character
   int charLocation = dataString.find( "+" );
   
   // search for next '+' character
   while ( charLocation < string::npos ) {
      dataString.replace( charLocation, 1, " " );
      charLocation = dataString.find( "+", charLocation + 1 );
   } // end while
   
   // find location of firstname
   int firstStart  = dataString.find( "firstname=" ) + 10;
   int endFirst = dataString.find( "&lastname" );
   
   firstname = dataString.substr( firstStart,
      endFirst - firstStart );
      
   // find location of lastname
   int lastStart = dataString.find( "lastname=" ) + 9;
   int endLast = dataString.find( "&address" );
   
   lastname = dataString.substr( lastStart, 
      endLast - lastStart );
   
   // find location of address
   int addressStart = dataString.find( "address=" ) + 8;
   int endAddress = dataString.find( "&town" );
   
   address = dataString.substr( addressStart,
      endAddress - addressStart );
   
   // find location of town
   int townStart = dataString.find( "town=" ) + 5;
   int endTown = dataString.find( "&state" );
   
   town = dataString.substr( townStart, endTown - townStart);
   
   // find location of state
   int stateStart = dataString.find( "state=" ) + 6;
   int endState = dataString.find( "&zipcode" );
   
   state = dataString.substr( stateStart, 
      endState - stateStart );
      
   // find location of zip code
   int zipStart = dataString.find( "zipcode=" ) + 8;
   int endZip = dataString.find( "&country" );
   
   zipcode = dataString.substr( zipStart, endZip - zipStart );
   
   // find location of country   
   int countryStart = dataString.find( "country=" ) + 8;
   int endCountry = dataString.find( "&email" );
   
   country = dataString.substr( countryStart,
      endCountry - countryStart );
   
   // find location of e-mail address
   int emailStart = dataString.find( "email=" ) + 6;
   int endEmail = dataString.find( "&submit" );
   
   email = dataString.substr( emailStart, 
      endEmail - emailStart );
   
   // output header
   cout << "Content-type: text/html\n\n";
   
   // output XML declaration and DOCTYPE
   cout << "<?xml version = \"1.0\"?>"
        << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 "
        << "Transitional//EN\" \"http://www.w3.org/TR/xhtml1"
        << "/DTD/xhtml1-transitional.dtd\">";

   // output html element and some of its contents
   cout << "<html xmlns = \"http://www.w3.org/1999/xhtml\">"
        << "<head><title>Contact Information entered"
        << "</title></head><body>";
   
   // output to file
   ofstream outFile( "clients.txt", ios::app );
   
   // file was not opened properly
   if ( !outFile ) {
      cerr << "Error: could not open contact file.";
      exit( 1 );
   } // end if
   
   // append data to clients.txt file
   outFile << firstname << " " << lastname << "\n"
           << address << "\n" << town << " "
           << state << " " << country << " "
           << zipcode << "\n" << email
           << "\n\n";
   
   // output data to user
   cout << "<table><tbody>"
        << "<tr><td>First Name:</td><td>"
        << firstname << "</td></tr>"
        << "<tr><td>Last Name:</td><td>"
        << lastname << "</td></tr>"
        << "<tr><td>Address:</td><td>"
        << address << "</td></tr>"
        << "<tr><td>Town:</td><td>"
        << town << "</td></tr>"
        << "<tr><td>State:</td><td>"
        << state << "</td></tr>"
        << "<tr><td>Zip Code:</td><td>"
        << zipcode << "</td></tr>"
        << "<tr><td>Country:</td><td>"
        << country << "</td></tr>"
        << "<tr><td>Email:</td><td>"
        << email << "</td></tr>"
        << "</tbody></table>"
        << "</body>\n</html>\n";
   
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
