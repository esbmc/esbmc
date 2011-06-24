%{

/*
 * This parser is specified based on:
 *
 * c5.y, a ANSI-C grammar written by James A. Roskind.
 * "Portions Copyright (c) 1989, 1990 James A. Roskind".
 * (http://www.idiom.com/free-compilers/,
 * ftp://ftp.infoseek.com/ftp/pub/c++grammar/,
 * ftp://ftp.sra.co.jp/.a/pub/cmd/c++grammar2.0.tar.gz)
 */

#define PARSER ansi_c_parser

#include "ansi_c_parser.h"

int yyansi_clex();
extern char *yyansi_ctext;

#include "parser_static.inc"

#include "y.tab.h"

/*** token declaration **************************************************/
%}

/*** ANSI-C keywords ***/

%token	TOK_AUTO      "auto"
%token  TOK_BOOL      "bool"
%token	TOK_BREAK     "break"
%token	TOK_CASE      "case"
%token	TOK_CHAR      "char"
%token	TOK_CONST     "const"
%token	TOK_CONTINUE  "continue"
%token	TOK_DEFAULT   "default"
%token	TOK_DO        "do"
%token	TOK_DOUBLE    "double"
%token	TOK_ELSE      "else"
%token	TOK_ENUM      "enum"
%token	TOK_EXTERN    "extern"
%token	TOK_FLOAT     "float"
%token	TOK_FOR       "for"
%token	TOK_GOTO      "goto"
%token	TOK_IF        "if"
%token  TOK_INLINE    "inline"
%token	TOK_INT       "int"
%token	TOK_LONG      "long"
%token	TOK_REGISTER  "register"
%token	TOK_RETURN    "return"
%token	TOK_SHORT     "short"
%token	TOK_SIGNED    "signed"
%token	TOK_SIZEOF    "sizeof"
%token	TOK_STATIC    "static"
%token	TOK_STRUCT    "struct"
%token	TOK_SWITCH    "switch"
%token	TOK_TYPEDEF   "typedef"
%token	TOK_UNION     "union"
%token	TOK_UNSIGNED  "unsigned"
%token	TOK_VOID      "void"
%token	TOK_VOLATILE  "volatile"
%token	TOK_WHILE     "while"

/*** multi-character operators ***/

%token	TOK_ARROW
%token	TOK_INCR
%token	TOK_DECR
%token	TOK_SHIFTLEFT
%token	TOK_SHIFTRIGHT
%token	TOK_LE
%token	TOK_GE
%token	TOK_EQ
%token	TOK_NE
%token	TOK_ANDAND
%token	TOK_OROR
%token	TOK_ELLIPSIS

/*** modifying assignment operators ***/

%token	TOK_MULTASSIGN
%token	TOK_DIVASSIGN
%token	TOK_MODASSIGN
%token	TOK_PLUSASSIGN
%token	TOK_MINUSASSIGN
%token	TOK_SLASSIGN
%token	TOK_SRASSIGN
%token	TOK_ANDASSIGN
%token	TOK_EORASSIGN
%token	TOK_ORASSIGN

/*** scanner parsed tokens (these have a value!) ***/

%token	TOK_IDENTIFIER
%token	TOK_TYPEDEFNAME
%token	TOK_INTEGER
%token	TOK_FLOATING
%token	TOK_CHARACTER
%token	TOK_STRING

/*** extensions ***/

%token	TOK_INT8
%token	TOK_INT16
%token	TOK_INT32
%token	TOK_INT64
%token	TOK_PTR32
%token	TOK_PTR64
%token  TOK_TYPEOF
%token  TOK_GCC_ASM
%token  TOK_MSC_ASM
%token	TOK_BUILTIN_VA_ARG
%token	TOK_BUILTIN_OFFSETOF

/*** special scanner reports ***/

%token	TOK_SCANNER_ERROR	/* used by scanner to report errors */
%token	TOK_SCANNER_EOF		/* used by scanner to report end of import */

/*** grammar selection ***/

%token	TOK_PARSE_LANGUAGE
%token	TOK_PARSE_EXPRESSION
%token	TOK_PARSE_TYPE

/*** priority, associativity, etc. definitions **************************/


%start	grammar

%expect 3	/* the famous "dangling `else'" ambiguity */
		/* results in one shift/reduce conflict   */
		/* that we don't want to be reported      */
		/* PLUS +2: KnR ambiguity */

%{
/************************************************************************/
/*** rules **************************************************************/
/************************************************************************/
%}
%%

/*** Grammar selection **************************************************/

grammar: TOK_PARSE_LANGUAGE translation_unit
	| TOK_PARSE_EXPRESSION comma_expression
	{
	  PARSER.parse_tree.declarations.push_back(ansi_c_declarationt());
	  PARSER.parse_tree.declarations.back().swap($2.expr);
	}
	| TOK_PARSE_TYPE type_name
	;

/*** Token with values **************************************************/


identifier:
	TOK_IDENTIFIER
	;

typedef_name:
	TOK_TYPEDEFNAME
	;

integer:
	TOK_INTEGER
	;

floating:
	TOK_FLOATING
	;

character:
	TOK_CHARACTER
	;

string:
	TOK_STRING
	;

/*** Constants **********************************************************/

/* note: the following has been changed from the ANSI-C grammar:	*/
/*	- constant includes string_literal_list (cleaner)		*/

constant:
	integer
	| floating
	| character
	| string_literal_list
	;

string_literal_list:
	string
	| string_literal_list string
	{ $$.expr = $1.expr;
	  // do concatenation
	  $$.expr.set("value", $$.expr.get_string("value")+
	    $2.expr.get_string("value"));
	}
	;

/*** Expressions ********************************************************/

primary_expression:
	identifier
	| constant
	| '(' comma_expression ')'
	{ $$.expr = $2.expr; }
	| statement_expression
	| builtin_va_arg_expression
	| builtin_offsetof
	;

builtin_va_arg_expression:
	TOK_BUILTIN_VA_ARG '(' assignment_expression ',' type_name ')'
	{
	  $$.expr=$1.expr;
	  $$.expr.id("builtin_va_arg");
	  mto($$.expr, $3.expr);
	  $$.expr.type().swap($5.expr);
	}
	;

builtin_offsetof:
	TOK_BUILTIN_OFFSETOF '(' type_name ',' offsetof_member_designator ')'
	{
	  $$.expr=$1.expr;
	  $$.expr.id("builtin_offsetof");
	  $$.expr.add("offsetof_type").swap($3.expr);
	  $$.expr.add("member").swap($5.expr);
	}
	;

offsetof_member_designator:
          member_name
        | offsetof_member_designator '.' member_name
        | offsetof_member_designator '[' comma_expression ']'
        ;                  

statement_expression: '(' compound_statement ')'
	{ $$.expr.id("sideeffect");
	  $$.expr.set("statement", "statement_expression");
          mto($$.expr, $2.expr);
	}
	;

postfix_expression:
	primary_expression
	| postfix_expression '[' comma_expression ']'
	{ binary($$.expr, $1.expr, $2.expr, "index", $3.expr); }
	| postfix_expression '(' ')'
	{ $$.expr=$2.expr;
	  set($$.expr, "sideeffect");
	  $$.expr.operands().resize(2);
	  $$.expr.op0().swap($1.expr);
	  $$.expr.op1().clear();
	  $$.expr.op1().id("arguments");
	  $$.expr.set("statement", "function_call");
	}
	| postfix_expression '(' argument_expression_list ')'
	{ $$.expr=$2.expr;
	  $$.expr.id("sideeffect");
	  $$.expr.set("statement", "function_call");
	  $$.expr.operands().resize(2);
	  $$.expr.op0().swap($1.expr);
	  $$.expr.op1().swap($3.expr);
	  $$.expr.op1().id("arguments");
	}
	| postfix_expression '.' member_name
	{ $$.expr=$2.expr;
	  set($$.expr, "member");
	  mto($$.expr, $1.expr);
	  $$.expr.set("component_name", $3.expr.get("#base_name"));
	}
	| postfix_expression TOK_ARROW member_name
	{ $$.expr=$2.expr;
	  set($$.expr, "ptrmember");
	  mto($$.expr, $1.expr);
	  $$.expr.set("component_name", $3.expr.get("#base_name"));
	}
	| postfix_expression TOK_INCR
	{ $$.expr=$2.expr;
	  $$.expr.id("sideeffect");
	  mto($$.expr, $1.expr);
	  $$.expr.set("statement", "postincrement");
	}
	| postfix_expression TOK_DECR
	{ $$.expr=$2.expr;
	  $$.expr.id("sideeffect");
	  mto($$.expr, $1.expr);
	  $$.expr.set("statement", "postdecrement");
	}
	;

member_name:
	identifier
	| typedef_name
	;

argument_expression_list:
	assignment_expression
	{
	  $$.expr.id("expression_list");
	  mto($$.expr, $1.expr);
	}
	| argument_expression_list ',' assignment_expression
	{
	  $$.expr=$1.expr;
	  mto($$.expr, $3.expr);
	}
	;

unary_expression:
	postfix_expression
	| TOK_INCR unary_expression
	{ $$.expr=$1.expr;
	  set($$.expr, "sideeffect");
	  $$.expr.set("statement", "preincrement");
	  mto($$.expr, $2.expr);
	}
	| TOK_DECR unary_expression
	{ $$.expr=$1.expr;
	  set($$.expr, "sideeffect");
	  $$.expr.set("statement", "predecrement");
	  mto($$.expr, $2.expr);
	}
	| '&' cast_expression
	{ $$.expr=$1.expr;
	  set($$.expr, "address_of");
	  mto($$.expr, $2.expr);
	}
	| '*' cast_expression
	{ $$.expr=$1.expr;
	  set($$.expr, "dereference");
	  mto($$.expr, $2.expr);
	}
	| '+' cast_expression
	{ $$.expr=$1.expr;
	  set($$.expr, "unary+");
	  mto($$.expr, $2.expr);
	}
	| '-' cast_expression
	{ $$.expr=$1.expr;
	  set($$.expr, "unary-");
	  mto($$.expr, $2.expr);
	}
	| '~' cast_expression
	{ $$.expr=$1.expr;
	  set($$.expr, "bitnot");
	  mto($$.expr, $2.expr);
	}
	| '!' cast_expression
	{ $$.expr=$1.expr;
	  set($$.expr, "not");
	  mto($$.expr, $2.expr);
	}
	| TOK_SIZEOF unary_expression
	{ $$.expr=$1.expr;
	  set($$.expr, "sizeof");
	  mto($$.expr, $2.expr);
	}
	| TOK_SIZEOF '(' type_name ')'
	{ $$.expr=$1.expr;
	  set($$.expr, "sizeof");
	  $$.expr.add("sizeof-type").swap($3.expr);
	}
	;

cast_expression:
	unary_expression
	| '(' type_name ')' cast_expression
	{
	  $$.expr=$1.expr;
	  set($$.expr, "typecast");
	  mto($$.expr, $4.expr);
	  $$.expr.type().swap($2.expr);
	}
	/* The following is a GCC extension
	   to allow a 'temporary union' */
	| '(' type_name ')' '{' designated_initializer_list '}'
	{
	  exprt tmp("designated_list");
	  tmp.operands().swap($5.expr).operands();
	  $$.expr=$1.expr;
	  set($$.expr, "typecast");
	  $$.expr.move_to_operands(tmp);
	  $$.expr.type().swap($2.expr);
	}
	;

multiplicative_expression:
	cast_expression
	| multiplicative_expression '*' cast_expression
	{ binary($$.expr, $1.expr, $2.expr, "*", $3.expr); }
	| multiplicative_expression '/' cast_expression
	{ binary($$.expr, $1.expr, $2.expr, "/", $3.expr); }
	| multiplicative_expression '%' cast_expression
	{ binary($$.expr, $1.expr, $2.expr, "mod", $3.expr); }
	;

additive_expression:
	multiplicative_expression
	| additive_expression '+' multiplicative_expression
	{ binary($$.expr, $1.expr, $2.expr, "+", $3.expr); }
	| additive_expression '-' multiplicative_expression
	{ binary($$.expr, $1.expr, $2.expr, "-", $3.expr); }
	;

shift_expression:
	additive_expression
	| shift_expression TOK_SHIFTLEFT additive_expression
	{ binary($$.expr, $1.expr, $2.expr, "shl", $3.expr); }
	| shift_expression TOK_SHIFTRIGHT additive_expression
	{ binary($$.expr, $1.expr, $2.expr, "shr", $3.expr); }
	;

relational_expression:
	shift_expression
	| relational_expression '<' shift_expression
	{ binary($$.expr, $1.expr, $2.expr, "<", $3.expr); }
	| relational_expression '>' shift_expression
	{ binary($$.expr, $1.expr, $2.expr, ">", $3.expr); }
	| relational_expression TOK_LE shift_expression
	{ binary($$.expr, $1.expr, $2.expr, "<=", $3.expr); }
	| relational_expression TOK_GE shift_expression
	{ binary($$.expr, $1.expr, $2.expr, ">=", $3.expr); }
	;

equality_expression:
	relational_expression
	| equality_expression TOK_EQ relational_expression
	{ binary($$.expr, $1.expr, $2.expr, "=", $3.expr); }
	| equality_expression TOK_NE relational_expression
	{ binary($$.expr, $1.expr, $2.expr, "notequal", $3.expr); }
	;

and_expression:
	equality_expression
	| and_expression '&' equality_expression
	{ binary($$.expr, $1.expr, $2.expr, "bitand", $3.expr); }
	;

exclusive_or_expression:
	and_expression
	| exclusive_or_expression '^' and_expression
	{ binary($$.expr, $1.expr, $2.expr, "bitxor", $3.expr); }
	;

inclusive_or_expression:
	exclusive_or_expression
	| inclusive_or_expression '|' exclusive_or_expression
	{ binary($$.expr, $1.expr, $2.expr, "bitor", $3.expr); }
	;

logical_and_expression:
	inclusive_or_expression
	| logical_and_expression TOK_ANDAND inclusive_or_expression
	{ binary($$.expr, $1.expr, $2.expr, "and", $3.expr); }
	;

logical_or_expression:
	logical_and_expression
	| logical_or_expression TOK_OROR logical_and_expression
	{ binary($$.expr, $1.expr, $2.expr, "or", $3.expr); }
	;

conditional_expression:
	logical_or_expression
	| logical_or_expression '?' comma_expression ':' conditional_expression
	{ $$.expr=$2.expr;
	  $$.expr.id("if");
	  mto($$.expr, $1.expr);
	  mto($$.expr, $3.expr);
	  mto($$.expr, $5.expr);
	}
	| logical_or_expression '?' ':' conditional_expression
	{ $$.expr=$2.expr;
	  $$.expr.id("sideeffect");
	  $$.expr.set("statement", "gcc_conditional_expression");
	  mto($$.expr, $1.expr);
	  mto($$.expr, $4.expr);
	}
	;

assignment_expression:
	conditional_expression
	| cast_expression '=' assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign"); }
	| cast_expression TOK_MULTASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign*"); }
	| cast_expression TOK_DIVASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign_div"); }
	| cast_expression TOK_MODASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign_mod"); }
	| cast_expression TOK_PLUSASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign+"); }
	| cast_expression TOK_MINUSASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign-"); }
	| cast_expression TOK_SLASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign_shl"); }
	| cast_expression TOK_SRASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign_shr"); }
	| cast_expression TOK_ANDASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign_bitand"); }
	| cast_expression TOK_EORASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign_bitxor"); }
	| cast_expression TOK_ORASSIGN assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "sideeffect", $3.expr); $$.expr.set("statement", "assign_bitor"); }
	;

comma_expression:
	assignment_expression
	| comma_expression ',' assignment_expression
	{ binary($$.expr, $1.expr, $2.expr, "comma", $3.expr); }
	;

constant_expression:
	assignment_expression
	;

comma_expression_opt:
	/* nothing */
	{ $$.expr.make_nil(); }
	| comma_expression
	;

/*** Declarations *******************************************************/


declaration:
	declaration_specifier ';'
	{
	}
	| type_specifier ';'
	{
	}
	| declaring_list ';'
	| default_declaring_list ';'
	;

default_declaring_list:
	declaration_qualifier_list identifier_declarator
		{
		  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
		}
	initializer_opt
		{
		  $$.expr.add("type")=$1.expr;
		  decl_statement($$.expr, $3.expr, $4.expr);
		}
	| type_qualifier_list identifier_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	initializer_opt
	{
	  $$.expr.add("type")=$1.expr;
	  decl_statement($$.expr, $3.expr, $4.expr);
	}
	| default_declaring_list ',' identifier_declarator
		{
		  const irept &t=$1.expr.find("type");
		  PARSER.new_declaration(t, $3.expr, $$.expr);
		}
		initializer_opt
	{
	  $$.expr=$1.expr;
	  decl_statement($$.expr, $4.expr, $5.expr);
	}
	;

declaring_list:			/* DeclarationSpec */
	declaration_specifier declarator
		{
		  // the symbol has to be visible during initialization
		  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
		}
		initializer_opt
	{
	  $$.expr.add("type")=$1.expr;
	  decl_statement($$.expr, $3.expr, $4.expr);
	}
	| type_specifier declarator
		{
		  // the symbol has to be visible during initialization
		  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
		}
		initializer_opt
	{
	  $$.expr.add("type")=$1.expr;
	  decl_statement($$.expr, $3.expr, $4.expr);
	}
	| declaring_list ',' declarator
		{
		  const irept &t=$1.expr.find("type");
		  PARSER.new_declaration(t, $3.expr, $$.expr);
		}
		initializer_opt
	{
	  $$.expr=$1.expr;
	  decl_statement($$.expr, $4.expr, $5.expr);
	}
	;

declaration_specifier:
	basic_declaration_specifier
	| sue_declaration_specifier
	| typedef_declaration_specifier
	;

type_specifier:
	basic_type_specifier
	| sue_type_specifier
	| typedef_type_specifier
	| typeof_type_specifier
	;

declaration_qualifier_list:
	storage_class
	| type_qualifier_list storage_class
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| declaration_qualifier_list declaration_qualifier
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	;

type_qualifier_list:
	type_qualifier
	| type_qualifier_list type_qualifier
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	;

declaration_qualifier:
	storage_class
	| type_qualifier
	;

type_qualifier:
	TOK_CONST      { $$.expr=$1.expr; set($$.expr, "const"); }
	| TOK_VOLATILE { $$.expr=$1.expr; set($$.expr, "volatile"); }
	;

basic_declaration_specifier:
	declaration_qualifier_list basic_type_name
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| basic_type_specifier storage_class
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| basic_declaration_specifier declaration_qualifier
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| basic_declaration_specifier basic_type_name
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	};

basic_type_specifier:
	basic_type_name
	| type_qualifier_list basic_type_name
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| basic_type_specifier type_qualifier
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| basic_type_specifier basic_type_name
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	};

sue_declaration_specifier:
	declaration_qualifier_list elaborated_type_name
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| sue_type_specifier storage_class
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| sue_declaration_specifier declaration_qualifier
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	;

sue_type_specifier:
	elaborated_type_name
	| type_qualifier_list elaborated_type_name
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| sue_type_specifier type_qualifier
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	;

typedef_declaration_specifier:	/* DeclarationSpec */
	typedef_type_specifier storage_class
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| declaration_qualifier_list typedef_name
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| typedef_declaration_specifier declaration_qualifier
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	;

typedef_type_specifier:		/* Type */
	typedef_name
	| type_qualifier_list typedef_name
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	| typedef_type_specifier type_qualifier
	{
	  $$.expr=$1.expr;
	  merge_types($$.expr, $2.expr);
	}
	;

typeof_type_specifier:
	TOK_TYPEOF '(' comma_expression ')'
	{ $$.expr = $3.expr;
	  locationt location=$$.expr.location();
	  typet new_type("type_of");
	  new_type.subtype() = (typet &)($$.expr);
	  $$.expr.swap(new_type);
	  $$.expr.location()=location;
	  $$.expr.set("#is_expression", true);
	}
	| TOK_TYPEOF '(' ptr_type_specifier  ')'
	{ $$.expr = $3.expr;
	  locationt location=$$.expr.location();
	  typet new_type("type_of");
	  new_type.subtype() = (typet &)($$.expr);
	  $$.expr.swap(new_type);
	  $$.expr.location()=location;
	  $$.expr.set("#is_expression", false);
	}
	;

ptr_type_specifier:
	type_specifier
	| ptr_type_specifier '*'
	{ $$.expr = $1.expr;
	  locationt location=$$.expr.location();
	  typet new_type("pointer");
	  new_type.subtype() = (typet&) $$.expr;
	  $$.expr.swap(new_type);
	  $$.expr.location()=location;
	}
	;

storage_class:
	TOK_TYPEDEF    { $$.expr=$1.expr; set($$.expr, "typedef"); }
	| TOK_EXTERN   { $$.expr=$1.expr; set($$.expr, "extern"); }
	| TOK_STATIC   { $$.expr=$1.expr; set($$.expr, "static"); }
	| TOK_AUTO     { $$.expr=$1.expr; set($$.expr, "auto"); }
	| TOK_REGISTER { $$.expr=$1.expr; set($$.expr, "register"); }
	| TOK_INLINE   { $$.expr=$1.expr; set($$.expr, "inline"); }
	;

basic_type_name:
	TOK_INT        { $$.expr=$1.expr; set($$.expr, "int"); }
	| TOK_INT8     { $$.expr=$1.expr; set($$.expr, "int8"); }
	| TOK_INT16    { $$.expr=$1.expr; set($$.expr, "int16"); }
	| TOK_INT32    { $$.expr=$1.expr; set($$.expr, "int32"); }
	| TOK_INT64    { $$.expr=$1.expr; set($$.expr, "int64"); }
	| TOK_PTR32    { $$.expr=$1.expr; set($$.expr, "ptr32"); }
	| TOK_PTR64    { $$.expr=$1.expr; set($$.expr, "ptr64"); }
	| TOK_CHAR     { $$.expr=$1.expr; set($$.expr, "char"); }
	| TOK_SHORT    { $$.expr=$1.expr; set($$.expr, "short"); }
	| TOK_LONG     { $$.expr=$1.expr; set($$.expr, "long"); }
	| TOK_FLOAT    { $$.expr=$1.expr; set($$.expr, "float"); }
	| TOK_DOUBLE   { $$.expr=$1.expr; set($$.expr, "double"); }
	| TOK_SIGNED   { $$.expr=$1.expr; set($$.expr, "signed"); }
	| TOK_UNSIGNED { $$.expr=$1.expr; set($$.expr, "unsigned"); }
	| TOK_VOID     { $$.expr=$1.expr; set($$.expr, "empty"); }
	| TOK_BOOL     { $$.expr=$1.expr; set($$.expr, "bool"); }
	;

elaborated_type_name:
	aggregate_name
	| enum_name
	;

aggregate_name:
	aggregate_key
		{
		  // an anon struct
		  exprt symbol("symbol");

		  symbol.set("#base_name", PARSER.get_anon_name());

		  PARSER.new_declaration($1.expr, symbol, $$.expr, true);
		}
		'{' member_declaration_list_opt '}'
	{
	  typet &type=$2.expr.type();
	  type.add("components").get_sub().swap($4.expr.add("operands").get_sub());

	  // grab symbol
	  $$.expr.id("symbol");
	  $$.expr.set("identifier", $2.expr.get("name"));
	  $$.expr.location()=$2.expr.location();

	  PARSER.move_declaration($2.expr);
	}
	| aggregate_key identifier_or_typedef_name
		{
		  PARSER.new_declaration($1.expr, $2.expr, $$.expr, true);

		  exprt tmp($$.expr);
		  tmp.type().id("incomplete_"+tmp.type().id_string());
		  PARSER.move_declaration(tmp);
		}
		'{' member_declaration_list_opt '}'
	{
	  typet &type=$3.expr.type();
	  type.add("components").get_sub().swap($5.expr.add("operands").get_sub());

	  // grab symbol
	  $$.expr.id("symbol");
	  $$.expr.set("identifier", $3.expr.get("name"));
	  $$.expr.location()=$3.expr.location();

	  PARSER.move_declaration($3.expr);
	}
	| aggregate_key identifier_or_typedef_name
	{
	  do_tag($1.expr, $2.expr);
	  $$.expr=$2.expr;
	}
	;

aggregate_key:
	TOK_STRUCT
	{ $$.expr=$1.expr; set($$.expr, "struct"); }
	| TOK_UNION
	{ $$.expr=$1.expr; set($$.expr, "union"); }
	;

member_declaration_list_opt:
		  /* Nothing */
	{
	  $$.expr.id("declaration_list");
	}
	| member_declaration_list
	;

member_declaration_list:
	  member_declaration
	| member_declaration_list member_declaration
	{
	  assert($1.expr.id()=="declaration_list");
	  assert($2.expr.id()=="declaration_list");
	  $$.expr=$1.expr;
	  Forall_operands(it, $2.expr)
	    $$.expr.move_to_operands(*it);
	  $2.expr.clear();
	}
	;

member_declaration:
	member_declaring_list ';'
	| member_default_declaring_list ';'
	| ';' /* empty declaration */
	{
	  $$.expr.id("declaration_list");
	}
	;

member_default_declaring_list:
	type_qualifier_list member_identifier_declarator
	{
	  $$.expr.id("declaration_list");

	  exprt declaration;

	  PARSER.new_declaration($1.expr, $2.expr, declaration, false, false);

	  $$.expr.move_to_operands(declaration);
	}
	| member_default_declaring_list ',' member_identifier_declarator
	{
	  exprt declaration;

	  typet type;
	  PARSER.new_declaration($1.expr, $3.expr, declaration, false, false);

	  $$.expr=$1.expr;
	  $$.expr.move_to_operands(declaration);
	}
	;

member_declaring_list:
	type_specifier member_declarator
	{
	  $$.expr.id("declaration_list");

	  // save the type_specifier
	  $$.expr.add("declaration_type")=$1.expr;

	  exprt declaration;
	  PARSER.new_declaration($1.expr, $2.expr, declaration, false, false);

	  $$.expr.move_to_operands(declaration);
	}
	| member_declaring_list ',' member_declarator
	{
	  exprt declaration;

	  irept declaration_type($1.expr.find("declaration_type"));
	  PARSER.new_declaration(declaration_type, $3.expr, declaration, false, false);

	  $$.expr=$1.expr;
	  $$.expr.move_to_operands(declaration);
	}
	;

member_declarator:
	declarator bit_field_size_opt
	{
	  if(!$2.expr.is_nil())
	  {
	    $$.expr=$2.expr;
	    $$.expr.add("subtype").swap($1.expr);
	  }
	  else
	    $$.expr=$1.expr;
	}
	| /* empty */
	{
	  $$.expr.make_nil();
	}
	| bit_field_size
	{
	  $$.expr=$1.expr;
	  $$.expr.add("subtype").make_nil();
	}
	;

member_identifier_declarator:
	identifier_declarator
		{ /* note: this mid-rule action (suggested by the grammar) */
		  /*       is not used because we don't have direct access */
		  /*       to the declaration specifier; therefore the     */
		  /*       symbol table is not updated ASAP (which is only */
		  /*       a minor problem; bit_field_size_opt expression  */
		  /*       cannot use the identifier_declarator)           */
		 }
		bit_field_size_opt
	{
	  $$.expr=$1.expr;
	  if(!$3.expr.is_nil())
	    merge_types($$.expr, $3.expr);
	}
	| bit_field_size
	{
	  // TODO
	  assert(0);
	}
	;

bit_field_size_opt:
	/* nothing */
	{
	  $$.expr.make_nil();
	}
	| bit_field_size
	;

bit_field_size:			/* Expression */
	':' constant_expression
	{
	  $$.expr=$1.expr; set($$.expr, "c_bitfield");
	  $$.expr.set("size", $2.expr);
	}
	;

/* note: although the grammar didn't suggest mid-rule actions here	*/
/*       we handle enum exactly like struct/union			*/
enum_name:			/* Type */
	enum_key
		{
		  // an anon enum
		  exprt symbol("symbol");
		  symbol.set("#base_name", PARSER.get_anon_name());

		  PARSER.new_declaration($1.expr, symbol, $$.expr, true);

		  exprt tmp($$.expr);
		  PARSER.move_declaration(tmp);
		}
		'{' enumerator_list '}'
	{
	  // grab symbol
	  $$.expr.id("symbol");
	  $$.expr.set("identifier", $2.expr.get("name"));
	  $$.expr.location()=$2.expr.location();

	  do_enum_members((const typet &)$$.expr, $4.expr);

	  PARSER.move_declaration($2.expr);
	}
	| enum_key identifier_or_typedef_name
		{ /* !!! mid-rule action !!! */
		  PARSER.new_declaration($1.expr, $2.expr, $$.expr, true);

		  exprt tmp($$.expr);
		  PARSER.move_declaration(tmp);
		}
		'{' enumerator_list '}'
	{
	  // grab symbol
	  $$.expr.id("symbol");
	  $$.expr.set("identifier", $3.expr.get("name"));
	  $$.expr.location()=$3.expr.location();

	  do_enum_members((const typet &)$$.expr, $5.expr);

	  PARSER.move_declaration($3.expr);
	}
	| enum_key identifier_or_typedef_name
	{
	  do_tag($1.expr, $2.expr);
	  $$.expr=$2.expr;
	}
	;

enum_key: TOK_ENUM
	{
	  $$.expr=$1.expr;
	  set($$.expr, "c_enum");
	}
	;

enumerator_list:		/* MemberList */
	enumerator_declaration
	{
	  mto($$.expr, $1.expr);
	}
	| enumerator_list ',' enumerator_declaration
	{
	  $$.expr=$1.expr;
	  mto($$.expr, $3.expr);
	}
	| enumerator_list ','
	{
	  $$.expr=$1.expr;
	}
	;

enumerator_declaration:
	  identifier_or_typedef_name enumerator_value_opt
	{
	  irept type("enum");
	  PARSER.new_declaration(type, $1.expr, $$.expr);
	  $$.expr.set("is_macro", true);
	  $$.expr.add("value").swap($2.expr);
	}
	;

enumerator_value_opt:		/* Expression */
	/* nothing */
	{
	  $$.expr.make_nil();
	}
	| '=' constant_expression
	{
	  $$.expr=$2.expr;
	}
	;

parameter_type_list:		/* ParameterList */
	parameter_list
	| parameter_list ',' TOK_ELLIPSIS
	{
	  typet tmp("ansi_c_ellipsis");
	  $$.expr=$1.expr;
	  ((typet &)$$.expr).move_to_subtypes(tmp);
	}
	| KnR_parameter_list
	;

KnR_parameter_list:
	KnR_parameter
	{
          $$.expr.id("arguments");
          mts($$.expr, $1.expr);
	}
	| KnR_parameter_list ',' KnR_parameter
	{
          $$.expr=$1.expr;
          mts($$.expr, $3.expr);
	}
	;

KnR_parameter: identifier
	{
	  irept type("KnR");
	  PARSER.new_declaration(type, $1.expr, $$.expr);
	}
	;

parameter_list:
	parameter_declaration
	{
	  $$.expr.id("arguments");
	  mts($$.expr, $1.expr);
	}
	| parameter_list ',' parameter_declaration
	{
	  $$.expr=$1.expr;
	  mts($$.expr, $3.expr);
	}
	;

parameter_declaration:
	declaration_specifier
	{
	  exprt nil;
	  nil.make_nil();
	  PARSER.new_declaration($1.expr, nil, $$.expr);
	}
	| declaration_specifier parameter_abstract_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	| declaration_specifier identifier_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	| declaration_specifier parameter_typedef_declarator
	{
          // the second tree is really the argument -- not part
          // of the type!
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	| declaration_qualifier_list
	{
	  exprt nil;
	  nil.make_nil();
	  PARSER.new_declaration($1.expr, nil, $$.expr);
	}
	| declaration_qualifier_list parameter_abstract_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	| declaration_qualifier_list identifier_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	| type_specifier
	{
	  exprt nil;
	  nil.make_nil();
	  PARSER.new_declaration($1.expr, nil, $$.expr);
	}
	| type_specifier parameter_abstract_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	| type_specifier identifier_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	| type_specifier parameter_typedef_declarator
	{
          // the second tree is really the argument -- not part
          // of the type!
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	| type_qualifier_list
	{
	  exprt nil;
	  nil.make_nil();
	  PARSER.new_declaration($1.expr, nil, $$.expr);
	}
	| type_qualifier_list parameter_abstract_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	| type_qualifier_list identifier_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	}
	;

identifier_or_typedef_name:
	identifier
	| typedef_name
	;

type_name:
	type_specifier
	| type_specifier abstract_declarator
	{
	  $$.expr=$1.expr;
	  make_subtype($$.expr, $2.expr);
	}
	| type_qualifier_list
	| type_qualifier_list abstract_declarator
	{
	  $$.expr=$1.expr;
	  make_subtype($$.expr, $2.expr);
	}
	;

initializer_opt:
	/* nothing */
	{
	  $$.expr.make_nil();
	}
	| '=' initializer
	{ $$.expr = $2.expr; }
	;

/* note: the following has been changed from the ANSI-C grammar:	*/
/*	- an initializer is not an assignment_expression,		*/
/*	  but a constant_expression					*/
/*	  (which probably is the case anyway for 99.9% of C programs)	*/

initializer:
	'{' initializer_list '}'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "constant");
	  $$.expr.type().id("incomplete_array");
	  $$.expr.operands().swap($2.expr.operands());
	}
	| '{' initializer_list ',' '}'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "constant");
	  $$.expr.type().id("incomplete_array");
	  $$.expr.operands().swap($2.expr.operands());
	}
	| constant_expression	/* was: assignment_expression */
	| '{' designated_initializer_list '}'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "designated_list");
	  $$.expr.operands().swap($2.expr.operands());
	}
	;

initializer_list:
	initializer
	{
	  $$.expr=$1.expr;
	  exprt tmp;
	  tmp.swap($$.expr);
	  $$.expr.clear();
	  $$.expr.move_to_operands(tmp);
	}
	| initializer_list ',' initializer
	{
	  $$.expr=$1.expr;
	  mto($$.expr, $3.expr);
	}
	;

/* GCC extension: designated initializer */
designated_initializer:
          /* empty */
        {
	  $$.expr.make_nil();          
        }
        | '.' identifier '=' initializer
        {
          $$.expr=$1.expr;
          $$.expr.id("designated_initializer");
          $$.expr.set("component_name", $2.expr.get("#base_name"));
          $$.expr.move_to_operands($4.expr);
        }
        ;

designated_initializer_list:
	designated_initializer
	{
	  $$.expr=$1.expr;
	  exprt tmp;
	  tmp.swap($$.expr);
	  $$.expr.clear();

	  if(tmp.is_not_nil())
            $$.expr.move_to_operands(tmp);
	}
	| designated_initializer_list ',' designated_initializer
	{
	  $$.expr=$1.expr;
	  if($3.expr.is_not_nil())
	    mto($$.expr, $3.expr);
	}
	;

/*** Statements *********************************************************/

statement:
	  labeled_statement
	| compound_statement
	| declaration_statement
	| expression_statement
	| selection_statement
	| iteration_statement
	| jump_statement
	| gcc_asm_statement
	| msc_asm_statement
	;

declaration_statement:
	declaration
	{
	  statement($$.expr, "decl-block");
	  $$.expr.operands().swap($1.expr.operands());
	}
	;

labeled_statement:
	identifier_or_typedef_name ':' statement
	{
	  $$.expr=$2.expr;
	  statement($$.expr, "label");
	  $$.expr.set("label", $1.expr.get("#base_name"));
	  mto($$.expr, $3.expr);
	}
	| TOK_CASE constant_expression ':' statement
	{
	  $$.expr=$1.expr;
	  statement($$.expr, "label");
	  mto($$.expr, $4.expr);
	  static_cast<exprt &>($$.expr.add("case")).
		move_to_operands($2.expr);
	}
	| TOK_DEFAULT ':' statement
	{
	  $$.expr=$1.expr;
	  statement($$.expr, "label");
	  mto($$.expr, $3.expr);
	  $$.expr.set("default", true);
	}
	;

/* note: the following has been changed from the ANSI-C grammar:	*/
/*	- rule compound_scope is used to prepare an inner scope for	*/
/*	  each compound_statement (and to obtain the line infos)	*/

compound_statement:
	compound_scope '{' '}'
	{
	  $$.expr=$2.expr;
	  statement($$.expr, "block");
	  $$.expr.set("#end_location", $3.expr.location());
	  PARSER.pop_scope();
	}
	| compound_scope '{' statement_list '}'
	{
	  $$.expr=$3.expr;
	  $$.expr.location()=$2.expr.location();
	  $$.expr.set("#end_location", $4.expr.location());
	  PARSER.pop_scope();
	}
	;

compound_scope:
	/* nothing */
	{
	  unsigned prefix=++PARSER.current_scope().compound_counter;
	  PARSER.new_scope(i2string(prefix)+"::");
	}
	;

statement_list:
	statement
	{
	  $$.expr=$1.expr;
	  to_code($$.expr).make_block();
	}
	| statement_list statement
	{
	  mto($$.expr, $2.expr);
	}
	;

expression_statement:
	comma_expression_opt ';'
	{
	  $$.expr=$2.expr;

	  if($1.expr.is_nil())
	    statement($$.expr, "skip");
	  else
	  {
	    statement($$.expr, "expression");
	    mto($$.expr, $1.expr);
	  }
	}
	;

selection_statement:
	  TOK_IF '(' comma_expression ')' statement
	{
	  $$.expr=$1.expr;
	  statement($$.expr, "ifthenelse");
	  mto($$.expr, $3.expr);
	  mto($$.expr, $5.expr);
	}
	| TOK_IF '(' comma_expression ')' statement TOK_ELSE statement
	{
	  $$.expr=$1.expr;
	  statement($$.expr, "ifthenelse");
	  mto($$.expr, $3.expr);
	  mto($$.expr, $5.expr);
	  mto($$.expr, $7.expr);
	}
	| TOK_SWITCH '(' comma_expression ')' statement
	{
	  $$.expr=$1.expr;
	  statement($$.expr, "switch");
	  mto($$.expr, $3.expr);
	  mto($$.expr, $5.expr);
	}
	;

declaration_or_expression_statement:
	  declaration_statement
	| expression_statement
	;

iteration_statement:
	TOK_WHILE '(' comma_expression_opt ')' statement
	{
	  $$.expr=$1.expr;
	  statement($$.expr, "while");
	  mto($$.expr, $3.expr);
	  mto($$.expr, $5.expr);
	}
	| TOK_DO statement TOK_WHILE '(' comma_expression ')' ';'
	{
	  $$.expr=$1.expr;
	  statement($$.expr, "dowhile");
	  mto($$.expr, $5.expr);
	  mto($$.expr, $2.expr);
	}
	| TOK_FOR '(' declaration_or_expression_statement
		comma_expression_opt ';' comma_expression_opt ')' statement
	{
	  $$.expr=$1.expr;
	  statement($$.expr, "for");
	  mto($$.expr, $3.expr);
	  mto($$.expr, $4.expr);
	  mto($$.expr, $6.expr);
	  mto($$.expr, $8.expr);
	}
	;

jump_statement:
	TOK_GOTO identifier_or_typedef_name ';'
	{
	  $$.expr=$1.expr;
	  statement($$.expr, "goto");
	  $$.expr.set("destination", $2.expr.get("#base_name"));
	}
	| TOK_CONTINUE ';'
	{ $$.expr=$1.expr; statement($$.expr, "continue"); }
	| TOK_BREAK ';'
	{ $$.expr=$1.expr; statement($$.expr, "break"); }
	| TOK_RETURN ';'
	{ $$.expr=$1.expr; statement($$.expr, "return"); }
	| TOK_RETURN comma_expression ';'
	{ $$.expr=$1.expr; statement($$.expr, "return"); mto($$.expr, $2.expr); }
	;

gcc_asm_statement:
	TOK_GCC_ASM volatile_opt '(' asm_commands ')' ';'
	{ $$.expr=$1.expr;
	  statement($$.expr, "asm");
	  $$.expr.set("flavor", "gcc"); }
	;

msc_asm_statement:
	TOK_MSC_ASM '{' TOK_STRING '}'
	{ $$.expr=$1.expr;
	  statement($$.expr, "asm"); 
	  $$.expr.set("flavor", "msc"); }
	| TOK_MSC_ASM TOK_STRING
	{ $$.expr=$1.expr;
	  statement($$.expr, "asm"); 
	  $$.expr.set("flavor", "msc"); }
	;

volatile_opt:
          /* nothing */
        | TOK_VOLATILE
        ;

/* asm ( assembler template
           : output operands                  // optional
           : input operands                   // optional
           : list of clobbered registers      // optional
           );
*/

asm_commands:
          asm_assembler_template
        | asm_assembler_template asm_outputs
        | asm_assembler_template asm_outputs asm_inputs
        | asm_assembler_template asm_outputs asm_inputs asm_clobbered_registers
	;

asm_assembler_template: string_literal_list
        ;

asm_outputs:
          ':' asm_output_list
        ;

asm_output:
          string '(' comma_expression ')'
        ;

asm_output_list:
          asm_output
        | asm_output_list ',' asm_output
        ;

asm_inputs:
          ':' asm_input_list
        ;

asm_input:
          string '(' comma_expression ')'
        ;

asm_input_list:
          asm_input
        | asm_input_list ',' asm_input

asm_clobbered_registers:
          ':' asm_clobbered_registers_list
        ;

asm_clobbered_registers_list:
          string
        | asm_clobbered_registers_list ',' string
        ;

/*** External Definitions ***********************************************/


/* note: the following has been changed from the ANSI-C grammar:	*/
/*	- translation unit is allowed to be empty!			*/

translation_unit:
	/* nothing */
	| external_definition_list
	;

external_definition_list:
	external_definition
	| external_definition_list external_definition
	;

external_definition:
	function_definition
	| declaration
	| ';' // empty declaration
	;

function_definition:
	function_head KnR_parameter_header_opt compound_statement
	{ 
          $1.expr.add("value").swap($3.expr);
          PARSER.pop_scope();
          PARSER.move_declaration($1.expr);
          PARSER.function="";
	}
	/* This is a GCC extension */
	| function_head KnR_parameter_header_opt gcc_asm_statement
	{ 
          // we ignore the value for now
          //$1.expr.add("value").swap($3.expr);
          PARSER.pop_scope();
          PARSER.move_declaration($1.expr);
          PARSER.function="";
	}
	;

KnR_parameter_header_opt:
          /* empty */
	| KnR_parameter_header
	;

KnR_parameter_header:
	  KnR_parameter_declaration
	| KnR_parameter_header KnR_parameter_declaration
	;

KnR_parameter_declaration: declaring_list ';'
	;

function_head:
	identifier_declarator /* void */
	{
	  irept type("int");
	  PARSER.new_declaration(type, $1.expr, $$.expr);
	  create_function_scope($$.expr);
	}
	| declaration_specifier declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	  create_function_scope($$.expr);
	}
	| type_specifier declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	  create_function_scope($$.expr);
	}
	| declaration_qualifier_list identifier_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	  create_function_scope($$.expr);
	}
	| type_qualifier_list identifier_declarator
	{
	  PARSER.new_declaration($1.expr, $2.expr, $$.expr);
	  create_function_scope($$.expr);
	}
	;

declarator:
	identifier_declarator
	| typedef_declarator
	;

typedef_declarator:
	paren_typedef_declarator
	| parameter_typedef_declarator
	;

parameter_typedef_declarator:
	typedef_name
	| typedef_name postfixing_abstract_declarator
	{
	  $$.expr=$1.expr;
	  make_subtype($$.expr, $2.expr);
	}
	| clean_typedef_declarator
	;

clean_typedef_declarator:	/* Declarator */
	clean_postfix_typedef_declarator
	| '*' parameter_typedef_declarator
	{
	  $$.expr=$2.expr;
	  do_pointer($1.expr, $2.expr);
	}
	| '*' type_qualifier_list parameter_typedef_declarator
	{
	  merge_types($2.expr, $3.expr);
	  $$.expr=$2.expr;
	  do_pointer($1.expr, $2.expr);
	}
	;

clean_postfix_typedef_declarator:	/* Declarator */
	'(' clean_typedef_declarator ')'
	{ $$.expr = $2.expr; }
	| '(' clean_typedef_declarator ')' postfixing_abstract_declarator
	{
	  /* note: this is a pointer ($2.expr) to a function ($4.expr) */
	  /* or an array ($4.expr)! */
	  $$.expr=$2.expr;
	  make_subtype($$.expr, $4.expr);
	}
	;

paren_typedef_declarator:	/* Declarator */
	paren_postfix_typedef_declarator
	| '*' '(' simple_paren_typedef_declarator ')'
	{
	  $$.expr=$3.expr;
	  do_pointer($1.expr, $3.expr);
	}
	| '*' type_qualifier_list '(' simple_paren_typedef_declarator ')'
	{
	  // not sure where the type qualifiers belong
	  merge_types($2.expr, $4.expr);
	  $$.expr=$2.expr;
	  do_pointer($1.expr, $2.expr);
	}
	| '*' paren_typedef_declarator
	{
	  $$.expr=$2.expr;
	  do_pointer($1.expr, $2.expr);
	}
	| '*' type_qualifier_list paren_typedef_declarator
	{
	  merge_types($2.expr, $3.expr);
	  $$.expr=$2.expr;
	  do_pointer($1.expr, $2.expr);
	}
	;

paren_postfix_typedef_declarator:	/* Declarator */
	'(' paren_typedef_declarator ')'
	{ $$.expr = $2.expr; }
	| '(' simple_paren_typedef_declarator postfixing_abstract_declarator ')'
	{	/* note: this is a function ($3.expr) with a typedef name ($2.expr) */
	  $$.expr=$2.expr;
	  make_subtype($$.expr, $3.expr);
	}
	| '(' paren_typedef_declarator ')' postfixing_abstract_declarator
	{
	  /* note: this is a pointer ($2.expr) to a function ($4.expr) */
	  /* or an array ($4.expr)! */
	  $$.expr=$2.expr;
	  make_subtype($$.expr, $4.expr);
	}
	;

simple_paren_typedef_declarator:
	typedef_name
	{
	  assert(0);
	}
	| '(' simple_paren_typedef_declarator ')'
	{ $$.expr = $2.expr; }
	;

identifier_declarator:
	  unary_identifier_declarator
	| paren_identifier_declarator
	;

unary_identifier_declarator:
	postfix_identifier_declarator
	| '*' identifier_declarator
	{
	  $$.expr=$2.expr;
	  do_pointer($1.expr, $2.expr);
	}
	| '*' type_qualifier_list identifier_declarator
	{
	  merge_types($2.expr, $3.expr);
	  $$.expr=$2.expr;
	  do_pointer($1.expr, $2.expr);
	}
	;

postfix_identifier_declarator:
        paren_identifier_declarator postfixing_abstract_declarator
	{
	  /* note: this is a function or array ($2.expr) with name ($1.expr) */
	  $$.expr=$1.expr;
	  make_subtype($$.expr, $2.expr);
	}
	| '(' unary_identifier_declarator ')'
	{ $$.expr = $2.expr; }
	| '(' unary_identifier_declarator ')' postfixing_abstract_declarator
	{
	  /* note: this is a pointer ($2.expr) to a function ($4.expr) */
	  /* or an array ($4.expr)! */
	  $$.expr=$2.expr;
	  make_subtype($$.expr, $4.expr);
	}
	;

paren_identifier_declarator:
	identifier
	| '(' paren_identifier_declarator ')'
	{ $$.expr=$2.expr; };

abstract_declarator:
	unary_abstract_declarator
	| postfix_abstract_declarator
	| postfixing_abstract_declarator
	;

parameter_abstract_declarator:
	parameter_unary_abstract_declarator
	| parameter_postfix_abstract_declarator
	;

postfixing_abstract_declarator:	/* AbstrDeclarator */
	array_abstract_declarator
	| '(' ')'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "code");
	  $$.expr.add("arguments");
	  $$.expr.add("subtype").make_nil();
	}
	| '('
	  {
		unsigned prefix=++PARSER.current_scope().compound_counter;
		PARSER.new_scope(i2string(prefix)+"::");
	  }
	  parameter_type_list ')'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "code");
	  $$.expr.add("subtype").make_nil();
	  $$.expr.add("arguments").get_sub().
	    swap($3.expr.add("subtypes").get_sub());
	  PARSER.pop_scope();
	}
	;

parameter_postfixing_abstract_declarator:
	array_abstract_declarator
	| '(' ')'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "code");
	  $$.expr.add("arguments");
	  $$.expr.add("subtype").make_nil();
	}
	| '('
	  {
		unsigned prefix=++PARSER.current_scope().compound_counter;
		PARSER.new_scope(i2string(prefix)+"::");
	  }
	  parameter_type_list ')'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "code");
	  $$.expr.add("subtype").make_nil();
	  $$.expr.add("arguments").get_sub().
	    swap($3.expr.add("subtypes").get_sub());
	  PARSER.pop_scope();
	}
	;

array_abstract_declarator:
	'[' ']'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "incomplete_array");
	  $$.expr.add("subtype").make_nil();
	}
	| '[' constant_expression ']'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "array");
	  $$.expr.add("size").swap($2.expr);
	  $$.expr.add("subtype").make_nil();
	}
	| array_abstract_declarator '[' constant_expression ']'
	{
	  // we need to push this down
	  $$.expr=$1.expr;
	  set($2.expr, "array");
	  $2.expr.add("size").swap($3.expr);
	  $2.expr.add("subtype").make_nil();
	  make_subtype($1.expr, $2.expr);
	}
	;

unary_abstract_declarator:
	'*'
	{
	  $$.expr=$1.expr;
	  set($$.expr, "pointer");
	  $$.expr.add("subtype").make_nil();
	}
	| '*' type_qualifier_list
	{
	  $$.expr=$2.expr;
	  exprt nil_declarator(static_cast<const exprt &>(get_nil_irep()));
	  merge_types($2.expr, nil_declarator);
	  do_pointer($1.expr, $2.expr);
	}
	| '*' abstract_declarator
	{
	  $$.expr=$2.expr;
	  do_pointer($1.expr, $2.expr);
	}
	| '*' type_qualifier_list abstract_declarator
	{
	  $$.expr=$2.expr;
	  merge_types($2.expr, $3.expr);
	  do_pointer($1.expr, $2.expr);
	}
	;

parameter_unary_abstract_declarator:
	'*'
	{
          $$.expr=$1.expr;
          set($$.expr, "pointer");
          $$.expr.add("subtype").make_nil();
	}
	| '*' type_qualifier_list
	{
          $$.expr=$2.expr;
          exprt nil_declarator(static_cast<const exprt &>(get_nil_irep()));
          merge_types($2.expr, nil_declarator);
          do_pointer($1.expr, $2.expr);
	}
	| '*' parameter_abstract_declarator
	{
          $$.expr=$2.expr;
          do_pointer($1.expr, $2.expr);
	}
	| '*' type_qualifier_list parameter_abstract_declarator
	{
          $$.expr=$2.expr;
          merge_types($2.expr, $3.expr);
          do_pointer($1.expr, $2.expr);
	}
	;

postfix_abstract_declarator:
	'(' unary_abstract_declarator ')'
	{ $$.expr = $2.expr; }
	| '(' postfix_abstract_declarator ')'
	{ $$.expr = $2.expr; }
	| '(' postfixing_abstract_declarator ')'
	{ $$.expr = $2.expr; }
	| '(' unary_abstract_declarator ')' postfixing_abstract_declarator
	{
	  /* note: this is a pointer ($2.expr) to a function ($4.expr) */
	  /* or an array ($4.expr) of pointers with name ($2.expr)! */
	  $$.expr=$2.expr;
	  make_subtype($$.expr, $4.expr);
	}
	;

parameter_postfix_abstract_declarator:
	'(' parameter_unary_abstract_declarator ')'
	{ $$.expr = $2.expr; }
	| '(' parameter_postfix_abstract_declarator ')'
	{ $$.expr = $2.expr; }
	| parameter_postfixing_abstract_declarator
	| '(' parameter_unary_abstract_declarator ')' parameter_postfixing_abstract_declarator
	{
	  /* note: this is a pointer ($2.expr) to a function ($4.expr) */
	  /* or an array ($4.expr) of pointers with name ($2.expr)! */
	  $$.expr=$2.expr;
	  make_subtype($$.expr, $4.expr);
	}
	;

%%
