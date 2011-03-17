%{

#include <stdio.h>
#include <string.h>

#include <expr.h>

#include "cvc_parser.h"

#define YYMAXDEPTH 20000
#define YYINITDEPTH 20000
#define YYSTYPE unsigned

#define mto(x, y) stack(x).move_to_operands(stack(y))

extern char *yycvctext;

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(exprt &expr)
{
  expr.clear();
  PARSER.set_location(expr);
}

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(YYSTYPE &expr)
 {
  newstack(expr);
  init(stack(expr));
 }

/*******************************************************************\

Function: init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void init(YYSTYPE &expr, const std::string &id)
 {
  init(expr);
  stack(expr).id(id);
 }

/*******************************************************************\

Function: binary_op

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void binary_op(YYSTYPE &dest,
                      YYSTYPE op1,
                      const std::string &id,
                      YYSTYPE op2)
 {
  init(dest, id);
  stack(dest).move_to_operands(stack(op1), stack(op2));
 }

/*******************************************************************\

Function: yycvcerror

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int yycvcerror(char *error)
{
  PARSER.parse_error(error, yycvctext);
  return strlen(error)+1;
}

int yylex();

%}

%token ASSERT
%token NEWLINE
%token STOP SKIP CHAOS
%token NAME NUMBER CSPTRUE CSPFALSE
%token CHANNEL DATATYPE NAMETYPE

%token OPEN CLOSE LSUBST RSUBST
%token LBRACE RBRACE LPBRACE RPBRACE EQUAL
%token COMMA DOTDOT PIPE BECOMES

%token UNION DIFF INTER
%token HEAD TAIL

%nonassoc LET IN LAMBDA LDOT IF THEN ELSE
%nonassoc AT COLON
%nonassoc LCOMM RCOMM LSQUARE PAR RSQUARE

%left BACKSLASH
%left INTL
%left NDET
%left BOX
%left INTR
%right SEMI 

%nonassoc WITHIN GUARD

%left DOT
%left OR
%left AND
%left NOT
%left EQ
%left NE
%left LT
%left LE
%left GT
%left GE
%left PLUS MINUS
%left TIMES SLASH MOD
%left HASH
%left CAT

%nonassoc ARROW
%right QUERY PLING

%%

defns	: 
	| defn
	| NEWLINE defns
	| defn NEWLINE defns
	;

defn	: lside EQUAL rside
	   {
	    init($$, "=");
            stack($$).move_to_operands(stack($1), stack($3));
	   }
	| CHANNEL names
	   {
            init($$, "channel");
           }
	| CHANNEL names COLON seq_type
           {
            init($$, "channel");
           }
        | DATATYPE name EQUAL datatype
           {
            init($$, "datatype"); 
           }
	;

newline0 :
	 | NEWLINE
	 ;

lside	: name
	| name OPEN exprs CLOSE { init($$, "unknown"); }
	;

rside	: proc
	| expr_minus_name
	;

seq_type: type
        | type DOT seq_type { init($$, "unknown"); }
        ;

type	: name
	| set
	;

datatype: name
        | name PIPE datatype { init($$, "unknown"); }
        ;

proc	: name
	| proc_minus_name
	;

proc_minus_name: OPEN proc_minus_name CLOSE { $$=$2; }
	| name OPEN exprs CLOSE	{ init($$, "unknown"); }
	| name ARROW proc   { binary_op($$, $1, "prefixed", $3); }
	| dotted ARROW proc { init($$, "unknown"); }
	| proc SEMI proc    { binary_op($$, $1, "sequential", $3); }
	| proc BOX proc	    { binary_op($$, $1, "external_choice", $3); }
	| proc NDET proc    { binary_op($$, $1, "internal_choice", $3); }
	| proc INTL proc    { init($$, "unknown"); }
	| proc LCOMM setname newline0 RCOMM proc { init($$, "unknown"); }
	| proc LSQUARE setname PAR setname RSQUARE proc	{ init($$, "unknown"); }
        | IF cond THEN proc ELSE proc
           {
            init($$, "ifthenelse");
            stack($$).move_to_operands(stack($2), stack($4), stack($6));
           }
	| STOP { init($$, "STOP"); }
	| SKIP { init($$, "SKIP"); }
        | CHAOS OPEN set CLOSE   { init($$, "unknown"); }
	| CHAOS OPEN name CLOSE  { init($$, "unknown"); }
	| proc BACKSLASH setname { binary_op($$, $1, "hide", $3); }
	| amb GUARD proc { init($$, "unknown"); }
	| NDET expr COLON setname AT proc	%prec AT { init($$, "unknown"); }
	| BOX  expr COLON setname AT proc	%prec AT { init($$, "unknown"); }
	| INTL expr COLON setname AT proc	%prec AT { init($$, "unknown"); }
	| SEMI expr COLON setname AT proc	%prec AT { init($$, "unknown"); }
	| PAR  expr COLON setname AT LSQUARE expr RSQUARE proc	%prec AT { init($$, "unknown"); }
	;

exprs	: expr { init($$); mto($$, $1); }
	| exprs COMMA expr { mto($1, $3); }
        ;

expr	: name
	| expr_minus_name
	;

expr_minus_name	: seq
		| bool
		| num
		| amb
		| tuple
		| set
		| commset
                | dotted
		;

dotted	: name dot_seq         { init($$, "unknown"); }
	| name COLON event_set { init($$, "unknown"); }
        ;

dot_seq	: DOT d_expr           { init($$, "unknown"); }
        | PLING d_expr         { init($$, "unknown"); }
	| QUERY d_expr         { init($$, "unknown"); }
        | DOT d_expr dot_seq   { init($$, "unknown"); }
	| PLING d_expr dot_seq { init($$, "unknown"); }
	| QUERY d_expr dot_seq { init($$, "unknown"); }
        ;

event_set: set
	| commset
	| name
	;

d_expr  : name
        | num
        | b_minus
        | set
        | commset
        | seq
        | tuple
        | amb
        ;

dotteds	: dotted { init($$); mto($$, $1); }
	| dotted COMMA dotted { mto($$, $1); }
	;

cond	: bool
	| amb
	;

bool	: b_minus
	| NOT expr      { init($$, "not"); mto($$, $2); }
	| expr AND expr	{ binary_op($$, $1, "and", $3); }
        | expr OR  expr { binary_op($$, $1, "or",  $3);  }
	| expr EQ  expr { binary_op($$, $1, "=",   $3);   }
	| expr NE  expr { binary_op($$, $1, "notequal", $3); }
	| expr LT  expr { binary_op($$, $1, "<",   $3);   }
	| expr GT  expr { binary_op($$, $1, ">",   $3);   }
	| expr LE  expr { binary_op($$, $1, "<=",  $3);  }
        | expr GE  expr { binary_op($$, $1, ">=",  $3);  }
	;

b_minus : CSPTRUE  { init($$); stack($$).make_true();  }
	| CSPFALSE { init($$); stack($$).make_false(); }
	;

num	: NUMBER
           {
            init($$, "constant");
            stack($$).set("value", stack($1).id());
            stack($$).type()=typet("integer");
           }
	| MINUS num_plus { init($$, "unary-"); mto($$, $2); }
	| HASH hash_arg { init($$, "unknown"); }
	| num_plus PLUS num_plus  { binary_op($$, $1, "+", $3); } 
	| num_plus MINUS num_plus { binary_op($$, $1, "-", $3); }
	| num_plus TIMES num_plus { binary_op($$, $1, "*", $3); }
	| num_plus SLASH num_plus { binary_op($$, $1, "/", $3); }
	| num_plus MOD num_plus	  { binary_op($$, $1, "%", $3); }
	;

num_plus: num
	| name
	| amb
	;

hash_arg: seq
        | set
        | tuple
        | commset
        | name
        | amb
        ;

name	: NAME { init($$, "name"); stack($$).set("identifier", stack($1).id()); }
	;

names	: name { init($$); mto($$, $1); }
	| name COMMA names { mto($$, $1); }
	;

tuple	: OPEN expr COMMA exprs CLOSE { init($$, "unknown"); }
	;

set	: LBRACE targ0 RBRACE { init($$, "unknown"); }
        | UNION OPEN setnames CLOSE { init($$, "unknown"); }
        | INTER OPEN setnames CLOSE { init($$, "unknown"); }
	| DIFF OPEN setname COMMA setname CLOSE { init($$, "unknown"); }
	;

setnames: setname { init($$); mto($$, $1); }
	| setname COMMA setnames { mto($$, $3); }
	;

setname	: set
        | commset
    	| name
        | amb
    	;

commset	: LPBRACE comm0 RPBRACE { init($$, "unknown"); }
	;

comm0	: 
	| names
	| dotteds
	;

seq	: LT targ0 GT { init($$, "unknown"); }
        | seqname CAT seqname { init($$, "unknown"); }
        | HEAD OPEN seqname CLOSE { init($$, "unknown"); }
        | TAIL OPEN seqname CLOSE { init($$, "unknown"); }
	;

seqname	: seq
	| name
	;

targ0	: { init($$); }
	| targs
	;

targs	: targ { init($$); mto($$, $1); }
	| targs COMMA targ { mto($$, $3); }
	;

targ	: b_minus
	| num
	| num_name DOTDOT num_name { binary_op($$, $1, "dotdot", $3); }
	| num_name DOTDOT { init($$, "unknown"); }
	| name
	| set
	| seq
	| tuple
	| dotted
	| amb
	;

num_name : num
         | name
         ;

amb	: OPEN expr CLOSE { $$=$2; }
	;
%%
