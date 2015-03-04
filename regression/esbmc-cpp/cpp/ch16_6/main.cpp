// Fig. 16.14: portal.cpp
// Handles entry to Bug2Bug Travel
#include <iostream>

using std::cout;
using std::cin;

#include <string>

using std::string;

#include <cstdlib>

int main()
{
   char postString[ 1024 ] = "";
   string dataString = "";
   string nameString = "";
   string passwordString = "";
   int contentLength = 0;
   
   // data was posted
   if ( getenv( "CONTENT_LENGTH" ) )
      contentLength = atoi( getenv( "CONTENT_LENGTH" ) );
   
   cin.read( postString, contentLength );
   dataString = postString;
   
   // search string for input data
   int namelocation = dataString.find( "namebox=" ) + 8;
   int endNamelocation = dataString.find( "&" );
   
   int password = dataString.find( "passwordbox=" ) + 12;
   int endPassword = dataString.find( "&button" );
   
   // get values for name and password
   nameString = dataString.substr( namelocation, 
      endNamelocation - namelocation );
   
   passwordString = dataString.substr( password, endPassword - 
      password );
   
   // output header
   cout << "Content-type: text/html\n\n";
   
   // output XML declaration and DOCTYPE
   cout << "<?xml version = \"1.0\"?>"
        << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 "
        << "Transitional//EN\" \"http://www.w3.org/TR/xhtml1"
        << "/DTD/xhtml1-transitional.dtd\">";
   
   // output html element and some of its contents
   cout << "<html xmlns = \"http://www.w3.org/1999/xhtml\">"
        << "<head><title>Bug2Bug Travel</title></head>"
        << "<body>";
   
   // output specials
   cout << "<h1>Welcome " << nameString << "!</h1>"
        << "<p>Here are our weekly specials:</p>"
        << "<ul><li>Boston to Taiwan ($875)</li>"
        << "<li>San Diego to Hong Kong ($750)</li>"
        << "<li>Chicago to Mexico City ($568)</li></ul>";
   
   // password is correct
   if ( passwordString == "coast2coast" )
      cout << "<hr /><p>Current member special: "
           << "Seattle to Tokyo ($400)</p>";
   
   // password was incorrect
   else
      cout << "<p>Sorry. You have entered an incorrect " 
           << "password</p>";
   
   cout << "</body></html>";
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
