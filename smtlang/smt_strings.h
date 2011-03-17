/*******************************************************************\
 *
 * Module: SMT-LIB constant strings
 *
 * Author: CM Wintersteiger 
 *
\*******************************************************************/

#ifndef _SMT_STRINGS_H_
#define _SMT_STRINGS_H_

#include <irep.h>

class smt_stringst
{
public:
  smt_stringst(void);
  
  const irep_idt strBOOL; 
  const irep_idt strLET;
  const irep_idt strFLET; 
  const irep_idt strVARIABLES;
  const irep_idt strVTERMS; 
  const irep_idt strVFORMULAS;
  const irep_idt strVAR; 
  const irep_idt strFVAR;
  const irep_idt strIDENTIFIER; 
  const irep_idt strVALUE;
  const irep_idt strTRUE; 
  const irep_idt strFALSE;
  const irep_idt strNOT; 
  const irep_idt strIMPL;
  const irep_idt strIFTHENELSE; 
  const irep_idt strAND;
  const irep_idt strOR; 
  const irep_idt strXOR;
  const irep_idt strIFF; 
  const irep_idt strSORTS;
  const irep_idt strSORT; 
  const irep_idt strITE;
  const irep_idt strFORALL; 
  const irep_idt strEXISTS;
  const irep_idt strQVARS; 
  const irep_idt strTYPE;
  const irep_idt strINDEX;  
  const irep_idt strNATURAL;
  const irep_idt strRATIONAL;   
  const irep_idt strIMPLIES;
  const irep_idt strArEQUAL; 
  const irep_idt strDISTINCT;
  const irep_idt emptySTR; 
  const irep_idt strINT;
  const irep_idt strREAL;
  const irep_idt strRETURNTYPE;
  const irep_idt strFUNCTIONS; 
  const irep_idt strPREDICATES;
  const irep_idt strANNOTATIONS;  
  const irep_idt strBVNAND; 
  const irep_idt strBVNOR;
  const irep_idt strBIT0; 
  const irep_idt strBIT1;
  const irep_idt strBVSLT; 
  const irep_idt strBVSLEQ;
  const irep_idt strBVSGT; 
  const irep_idt strBVSGEQ;
  const irep_idt strSIGNEXTEND; 
  const irep_idt strSHIFTLEFT0;
  const irep_idt strSHIFTRIGHT0; 
  const irep_idt strSHIFTLEFT1;  
  const irep_idt strSHIFTRIGHT1; 
  const irep_idt strREPEAT;
  const irep_idt strROTATELEFT; 
  const irep_idt strROTATERIGHT;    
};



#endif
