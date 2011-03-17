%option noyywrap
%option nounput 

%{
static int line   = 1;
static int column = 1;

#include <string.h>
#include <math.h>

#include "expr.h"
#include "smt_parser.h"

void FlexPosition();
unsigned yysmtlval;

#define DIM(a)  (sizeof(a)/sizeof((a)[0]))
#define WHITE_RETURN()   FlexPosition()
#define FlexReturn(tk) { FlexPosition(); return tk; }

%}

ident  [A-Za-z][A-Za-z\'0-9_\.]*
numeral (0)|([1-9][0-9]*)
arith [=<>&@#+\-*/%|~]+
ratio {numeral}"."0*{numeral}
index \[{numeral}(":"{numeral})*\]

%x ANNOTATION
%x INCURLY
%x INSTRING

%%

<ANNOTATION>"{"     { BEGIN(INCURLY); FlexReturn(LCB); }
<INCURLY>[^}\\]*(\\.[^}\\]*)*    { newstack(yysmtlval);
                                   stack(yysmtlval).id(yysmttext);
                                   BEGIN(0); FlexReturn(USER_VALUE_CONTENT); }                     
<ANNOTATION>"}"     { BEGIN(0); }

<INSTRING>"\""      { BEGIN(0); FlexReturn(QUOTE); }
<INSTRING>[^"\\]*(\\.[^"\\]*)*    { newstack(yysmtlval);
                      stack(yysmtlval).id(yysmttext); 
                      FlexReturn(STRING_CONTENT); }


                      
<*>[ \t\r\n]+ { WHITE_RETURN(); }
<*>";"[^\n]*  { WHITE_RETURN(); }
              
<*>"("    { FlexReturn(LPAR); }
<*>")"    { FlexReturn(RPAR); }
<*>"["    { FlexReturn(LBR); }
<*>"]"    { FlexReturn(RBR); }
<*>"{"    { FlexReturn(LCB); }
<*>"}"    { FlexReturn(RCB); }
<*>"\"" { BEGIN(INSTRING); FlexReturn(QUOTE); }
<*>"benchmark"  {FlexReturn(BENCHMARK); }
<*>"\:status" {FlexReturn(STATUS); }


<*>"sat" |
<*>"unsat" |
<*>"unknown" { newstack(yysmtlval);
               stack(yysmtlval).id(yysmttext); 
               FlexReturn(STATUS_ID); }
<*>"\:logic"      { FlexReturn(LOGICA); }
<*>"\:extrafuns"  { FlexReturn(EXTRAFUNS); }
<*>"\:extrapreds" { FlexReturn(EXTRAPREDS); }
<*>"\:extrasorts" { FlexReturn(EXTRASORTS); }
<*>"\:formula"    { FlexReturn(FORMULA); }
<*>"\:assumption" { FlexReturn(ASSUMPTION); }
<*>"\:notes"      { FlexReturn(NOTES); }

<*>"theory"       { FlexReturn(THEORY); }
<*>"logic"        { FlexReturn(LOGIC); }

<*>"\:theory"     { FlexReturn(THEORYA); }
<*>"\:sorts"      { FlexReturn(SORTS); }
<*>"\:funs"       { FlexReturn(FUNS); }
<*>"\:preds"      { FlexReturn(PREDS); }
<*>"\:definition" { FlexReturn(DEFINITION); }
<*>"\:axioms"     { FlexReturn(AXIOMS); }
<*>"\:language"   { FlexReturn(LANGUAGE); }
<*>"\:extensions" { FlexReturn(EXTENSIONS); }


<*>"forall"       { FlexReturn(FORALL); }
<*>"exists"       { FlexReturn(EXISTS); }
<*>"true"         { FlexReturn(TRUE); }
<*>"false"        { FlexReturn(FALSE); }
<*>"distinct"     { FlexReturn(DISTINCT); }
<*>"not"          { FlexReturn(NOT); }
<*>"implies"      { FlexReturn(IMPLIES); }
<*>"ite"          { FlexReturn(ITE); }
<*>"if_then_else" { FlexReturn(IF_THEN_ELSE); }
<*>"and"          { FlexReturn(AND); }
<*>"or"           { FlexReturn(OR); }
<*>"xor"          { FlexReturn(XOR); }
<*>"iff"          { FlexReturn(IFF); }
<*>"let"          { FlexReturn(LET); }
<*>"flet"         { FlexReturn(FLET); }
<*>"?"{ident} { newstack(yysmtlval);
                stack(yysmtlval).id(yysmttext);
                FlexReturn(VAR); }
<*>"$"{ident} { newstack(yysmtlval);
                stack(yysmtlval).id(yysmttext);
                FlexReturn (FVAR); }
<*>":"{ident} { newstack(yysmtlval);
                stack(yysmtlval).id(yysmttext);
                BEGIN(ANNOTATION); FlexReturn(ATTRIBUTE);}
<*>{numeral}  { newstack(yysmtlval);
                stack(yysmtlval).id("natural");
                stack(yysmtlval).set("value", yysmttext);
                FlexReturn (NUMERAL); }
<*>{ident}  { newstack(yysmtlval);
              stack(yysmtlval).id(yysmttext);
              FlexReturn (IDENT);}
<*>{arith}  { newstack(yysmtlval); 
              stack(yysmtlval).id(yysmttext);
              FlexReturn(ARITH_SYMB);}
<*>{ratio}  { newstack(yysmtlval); 
              stack(yysmtlval).id("rational");
              stack(yysmtlval).set("value", yysmttext);
              FlexReturn(RATIONAL);}
<*>{index}  { newstack(yysmtlval); 
              stack(yysmtlval).id("index");
              std::string val = yysmttext;
              val = val.substr(1, val.length()-2);
              stack(yysmtlval).set("value", val);              
              FlexReturn(INDEX);}

<<EOF>>     { FlexReturn(LEXEOF); }
%%

#include <stdio.h>
#include <ctype.h>

#define YYERRCODE       256

char  yyerrsrc[64] = "";                        /* current input file name */

void FlexPosition()
{
  int i;
   
  for(i=0; i<yyleng; i++)
  {
    if (yytext[i]=='\n')
    {
      column = 0;
      line++;
    }
    else
      column++;

    PARSER.column_no=column;
  }
}
