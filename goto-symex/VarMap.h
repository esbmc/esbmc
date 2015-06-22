//Bruno J. Savino

#ifndef VARMAP_H
#define VARMAP_H

#define LNUMBERSIZE 10
#define LINFOSIZE 100

#define VNAMESIZE 20
#define VINFOSIZE 20

#include <iostream>

class VarMap {

public:

   // default VarMap constructor
   VarMap( std::string = "", std::string = "", std::string = "", std::string = "");

   // accessor functions for LineNumber
   void setLineNumber( std::string );
   std::string getLineNumber() const;
   // accessor functions for LineInfo
   void setLineInfo( std::string );
   std::string getLineInfo() const;

   // accessor functions for VarName
   void setVarName( std::string );
   std::string getVarName() const;
   // accessor functions for VarInfo
   void setVarInfo( std::string );
   std::string getVarInfo() const;



private:
   char lineNumber[LNUMBERSIZE];
   char lineInfo[LINFOSIZE];

   char varName[VNAMESIZE];
   char varInfo[VINFOSIZE];

}; // end class VarMap

#endif
