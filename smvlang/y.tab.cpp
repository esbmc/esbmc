/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse yysmvparse
#define yylex   yysmvlex
#define yyerror yysmverror
#define yylval  yysmvlval
#define yychar  yysmvchar
#define yydebug yysmvdebug
#define yynerrs yysmvnerrs


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     AG_Token = 258,
     AX_Token = 259,
     AF_Token = 260,
     INIT_Token = 261,
     TRANS_Token = 262,
     SPEC_Token = 263,
     VAR_Token = 264,
     DEFINE_Token = 265,
     ASSIGN_Token = 266,
     INVAR_Token = 267,
     FAIRNESS_Token = 268,
     MODULE_Token = 269,
     ARRAY_Token = 270,
     OF_Token = 271,
     DOTDOT_Token = 272,
     BOOLEAN_Token = 273,
     EXTERN_Token = 274,
     NEXT_Token = 275,
     INC_Token = 276,
     DEC_Token = 277,
     CASE_Token = 278,
     ESAC_Token = 279,
     BECOMES_Token = 280,
     ADD_Token = 281,
     SUB_Token = 282,
     SWITCH_Token = 283,
     init_Token = 284,
     PLUS_Token = 285,
     STRING_Token = 286,
     QSTRING_Token = 287,
     QUOTE_Token = 288,
     NUMBER_Token = 289,
     IMPLIES_Token = 290,
     UNION_Token = 291,
     EQUIV_Token = 292,
     XOR_Token = 293,
     OR_Token = 294,
     AND_Token = 295,
     NOT_Token = 296,
     UNTIL_Token = 297,
     A_Token = 298,
     E_Token = 299,
     EG_Token = 300,
     EF_Token = 301,
     EX_Token = 302,
     GE_Token = 303,
     LE_Token = 304,
     GT_Token = 305,
     LT_Token = 306,
     NOTEQUAL_Token = 307,
     EQUAL_Token = 308,
     MINUS_Token = 309,
     DIVIDE_Token = 310,
     TIMES_Token = 311,
     UMINUS = 312,
     DOT_Token = 313
   };
#endif
/* Tokens.  */
#define AG_Token 258
#define AX_Token 259
#define AF_Token 260
#define INIT_Token 261
#define TRANS_Token 262
#define SPEC_Token 263
#define VAR_Token 264
#define DEFINE_Token 265
#define ASSIGN_Token 266
#define INVAR_Token 267
#define FAIRNESS_Token 268
#define MODULE_Token 269
#define ARRAY_Token 270
#define OF_Token 271
#define DOTDOT_Token 272
#define BOOLEAN_Token 273
#define EXTERN_Token 274
#define NEXT_Token 275
#define INC_Token 276
#define DEC_Token 277
#define CASE_Token 278
#define ESAC_Token 279
#define BECOMES_Token 280
#define ADD_Token 281
#define SUB_Token 282
#define SWITCH_Token 283
#define init_Token 284
#define PLUS_Token 285
#define STRING_Token 286
#define QSTRING_Token 287
#define QUOTE_Token 288
#define NUMBER_Token 289
#define IMPLIES_Token 290
#define UNION_Token 291
#define EQUIV_Token 292
#define XOR_Token 293
#define OR_Token 294
#define AND_Token 295
#define NOT_Token 296
#define UNTIL_Token 297
#define A_Token 298
#define E_Token 299
#define EG_Token 300
#define EF_Token 301
#define EX_Token 302
#define GE_Token 303
#define LE_Token 304
#define GT_Token 305
#define LT_Token 306
#define NOTEQUAL_Token 307
#define EQUAL_Token 308
#define MINUS_Token 309
#define DIVIDE_Token 310
#define TIMES_Token 311
#define UMINUS 312
#define DOT_Token 313




/* Copy the first part of user declarations.  */
#line 1 "parser.y"

#include <i2string.h>

#include "smv_parser.h"
#include "smv_typecheck.h"

#define YYSTYPE unsigned
#define PARSER smv_parser

#include "y.tab.h"

#define YYMAXDEPTH 200000
#define YYSTYPE_IS_TRIVIAL 1

/*------------------------------------------------------------------------*/

#define yylineno yysmvlineno
#define yytext yysmvtext

#define yyerror yysmverror
int yysmverror(const std::string &error);
int yylex();
extern char *yytext;

/*------------------------------------------------------------------------*/

#define mto(x, y) stack(x).move_to_operands(stack(y))
#define binary(x, y, id, z) { init(x, id); \
  stack(x).move_to_operands(stack(y), stack(z)); }

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

Function: mk_index

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
static void mk_index(YYSTYPE &dest, YYSTYPE &op, YYSTYPE &index)
{
  init(dest, "extractbit");
  stack(dest).set("index", stack(index));
  mto(dest, op);
}
#endif

/*******************************************************************\

Function: j_binary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void j_binary(YYSTYPE &dest, YYSTYPE &op1,
                     const std::string &id, YYSTYPE &op2)
{
  if(stack(op1).id()==id)
  {
    dest=op1;
    mto(dest, op2);
  }
  else if(stack(op2).id()==id)
  {
    dest=op2;
    mto(dest, op1);
  }
  else
    binary(dest, op1, id, op2);
}

/*******************************************************************\

Function: new_module

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void new_module(YYSTYPE &module)
{
  const std::string name=smv_module_symbol(stack(module).id_string());
  PARSER.module=&PARSER.parse_tree.modules[name];
  PARSER.module->name=name;
  PARSER.module->base_name=stack(module).id_string();
  PARSER.module->used=true;
}

/*------------------------------------------------------------------------*/



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 389 "y.tab.cpp"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  45
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   535

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  68
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  34
/* YYNRULES -- Number of rules.  */
#define YYNRULES  112
/* YYNRULES -- Number of states.  */
#define YYNSTATES  205

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   313

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      59,    60,     2,     2,    62,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    65,    61,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    66,     2,    67,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    63,     2,    64,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    12,    15,    17,    19,
      22,    23,    30,    31,    34,    37,    39,    42,    46,    48,
      51,    55,    57,    60,    64,    66,    69,    71,    74,    76,
      79,    83,    85,    88,    92,    94,    97,   101,   105,   107,
     110,   112,   114,   118,   119,   121,   128,   130,   134,   138,
     140,   142,   147,   149,   153,   155,   160,   162,   165,   167,
     170,   178,   180,   182,   184,   186,   189,   194,   196,   198,
     203,   207,   211,   216,   221,   228,   235,   237,   241,   249,
     252,   256,   260,   264,   268,   272,   276,   280,   283,   286,
     289,   292,   296,   300,   304,   308,   312,   316,   320,   322,
     326,   328,   330,   332,   336,   340,   345,   350,   351,   354,
     359,   360,   363
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      69,     0,    -1,    70,    -1,    93,    -1,    71,    -1,    70,
      71,    -1,    73,    75,    -1,    31,    -1,    33,    -1,    14,
      72,    -1,    -1,    14,    72,    74,    59,    81,    60,    -1,
      -1,    76,    75,    -1,     9,    78,    -1,     9,    -1,     6,
      93,    -1,     6,    93,    61,    -1,     6,    -1,     7,    93,
      -1,     7,    93,    61,    -1,     7,    -1,     8,    93,    -1,
       8,    93,    61,    -1,     8,    -1,    11,    87,    -1,    11,
      -1,    10,    91,    -1,    10,    -1,    12,    93,    -1,    12,
      93,    61,    -1,    12,    -1,    13,    93,    -1,    13,    93,
      61,    -1,    13,    -1,    19,    77,    -1,    19,    77,    61,
      -1,    96,    53,    33,    -1,    86,    -1,    86,    78,    -1,
      96,    -1,    79,    -1,    80,    62,    79,    -1,    -1,    80,
      -1,    15,    34,    17,    34,    16,    82,    -1,    18,    -1,
      63,    84,    64,    -1,    34,    17,    34,    -1,    83,    -1,
      72,    -1,    72,    59,    95,    60,    -1,    85,    -1,    84,
      62,    85,    -1,    31,    -1,    96,    65,    82,    61,    -1,
      88,    -1,    88,    87,    -1,    92,    -1,    92,    87,    -1,
      90,    59,    89,    60,    25,    93,    61,    -1,    96,    -1,
      29,    -1,    20,    -1,    92,    -1,    92,    91,    -1,    89,
      25,    93,    61,    -1,    94,    -1,    96,    -1,    20,    59,
      94,    60,    -1,    59,    93,    60,    -1,    63,    95,    64,
      -1,    21,    59,    94,    60,    -1,    22,    59,    94,    60,
      -1,    26,    59,    94,    62,    94,    60,    -1,    27,    59,
      94,    62,    94,    60,    -1,    34,    -1,    23,    98,    24,
      -1,    28,    59,    96,    60,    63,   100,    64,    -1,    54,
      94,    -1,    94,    30,    94,    -1,    94,    54,    94,    -1,
      94,    37,    94,    -1,    94,    35,    94,    -1,    94,    38,
      94,    -1,    94,    39,    94,    -1,    94,    40,    94,    -1,
      41,    94,    -1,     4,    94,    -1,     5,    94,    -1,     3,
      94,    -1,    94,    53,    94,    -1,    94,    52,    94,    -1,
      94,    51,    94,    -1,    94,    49,    94,    -1,    94,    50,
      94,    -1,    94,    48,    94,    -1,    94,    36,    94,    -1,
      93,    -1,    95,    62,    93,    -1,    97,    -1,    32,    -1,
      31,    -1,    97,    58,    32,    -1,    97,    58,    31,    -1,
      97,    66,    34,    67,    -1,    97,    59,    34,    60,    -1,
      -1,    98,    99,    -1,    93,    65,    93,    61,    -1,    -1,
     100,   101,    -1,    34,    65,    94,    61,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   188,   188,   189,   193,   194,   197,   200,   201,   204,
     205,   205,   208,   209,   212,   213,   214,   215,   216,   217,
     218,   219,   220,   221,   222,   223,   224,   225,   226,   227,
     228,   229,   230,   231,   232,   233,   234,   237,   252,   253,
     256,   265,   266,   269,   270,   273,   289,   290,   291,   297,
     300,   306,   315,   320,   327,   334,   367,   368,   369,   370,
     373,   391,   394,   395,   398,   399,   402,   437,   440,   441,
     442,   443,   444,   445,   446,   447,   448,   449,   450,   451,
     452,   453,   454,   455,   456,   457,   458,   459,   460,   461,
     462,   463,   464,   465,   466,   467,   468,   469,   472,   473,
     476,   505,   509,   510,   517,   524,   532,   542,   543,   546,
     549,   550,   553
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "AG_Token", "AX_Token", "AF_Token",
  "INIT_Token", "TRANS_Token", "SPEC_Token", "VAR_Token", "DEFINE_Token",
  "ASSIGN_Token", "INVAR_Token", "FAIRNESS_Token", "MODULE_Token",
  "ARRAY_Token", "OF_Token", "DOTDOT_Token", "BOOLEAN_Token",
  "EXTERN_Token", "NEXT_Token", "INC_Token", "DEC_Token", "CASE_Token",
  "ESAC_Token", "BECOMES_Token", "ADD_Token", "SUB_Token", "SWITCH_Token",
  "init_Token", "PLUS_Token", "STRING_Token", "QSTRING_Token",
  "QUOTE_Token", "NUMBER_Token", "IMPLIES_Token", "UNION_Token",
  "EQUIV_Token", "XOR_Token", "OR_Token", "AND_Token", "NOT_Token",
  "UNTIL_Token", "A_Token", "E_Token", "EG_Token", "EF_Token", "EX_Token",
  "GE_Token", "LE_Token", "GT_Token", "LT_Token", "NOTEQUAL_Token",
  "EQUAL_Token", "MINUS_Token", "DIVIDE_Token", "TIMES_Token", "UMINUS",
  "DOT_Token", "'('", "')'", "';'", "','", "'{'", "'}'", "':'", "'['",
  "']'", "$accept", "start", "modules", "module", "module_name",
  "module_head", "@1", "sections", "section", "extern_var", "vardecls",
  "module_argument", "module_argument_list", "module_argument_list_opt",
  "type", "usertype", "enum_list", "enum_element", "vardecl",
  "assignments", "assignment", "assignment_var", "assignment_head",
  "defines", "define", "formula", "term", "formula_list", "variable_name",
  "qstring_list", "cases", "case", "switches", "switch", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,    40,
      41,    59,    44,   123,   125,    58,    91,    93
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    68,    69,    69,    70,    70,    71,    72,    72,    73,
      74,    73,    75,    75,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    76,    76,    76,    76,    77,    78,    78,
      79,    80,    80,    81,    81,    82,    82,    82,    82,    82,
      83,    83,    84,    84,    85,    86,    87,    87,    87,    87,
      88,    89,    90,    90,    91,    91,    92,    93,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    95,    95,
      96,    97,    97,    97,    97,    97,    97,    98,    98,    99,
     100,   100,   101
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     2,     2,     1,     1,     2,
       0,     6,     0,     2,     2,     1,     2,     3,     1,     2,
       3,     1,     2,     3,     1,     2,     1,     2,     1,     2,
       3,     1,     2,     3,     1,     2,     3,     3,     1,     2,
       1,     1,     3,     0,     1,     6,     1,     3,     3,     1,
       1,     4,     1,     3,     1,     4,     1,     2,     1,     2,
       7,     1,     1,     1,     1,     2,     4,     1,     1,     4,
       3,     3,     4,     4,     6,     6,     1,     3,     7,     2,
       3,     3,     3,     3,     3,     3,     3,     2,     2,     2,
       2,     3,     3,     3,     3,     3,     3,     3,     1,     3,
       1,     1,     1,     3,     3,     4,     4,     0,     2,     4,
       0,     2,     4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,   107,     0,
       0,     0,   102,   101,    76,     0,     0,     0,     0,     0,
       2,     4,    12,     3,    67,    68,   100,    90,    88,    89,
       7,     8,     9,     0,     0,     0,     0,     0,     0,     0,
      87,    79,     0,    98,     0,     1,     5,    18,    21,    24,
      15,    28,    26,    31,    34,     0,     6,    12,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    77,
       0,   108,     0,     0,     0,    70,     0,    71,    16,    19,
      22,    14,    38,     0,     0,    27,    64,    61,    63,    62,
      25,    56,     0,    58,    29,    32,    35,     0,    13,    80,
      83,    97,    82,    84,    85,    86,    96,    94,    95,    93,
      92,    91,    81,   104,   103,     0,     0,    43,    69,    72,
      73,     0,     0,     0,     0,    99,    17,    20,    23,    39,
       0,     0,    65,    57,     0,    59,    30,    33,    36,     0,
     106,   105,    41,    44,     0,    40,     0,     0,     0,   110,
       0,    46,     0,     0,    50,     0,    49,     0,     0,    37,
       0,    11,   109,    74,    75,     0,     0,     0,    54,     0,
      52,     0,    55,    66,     0,    42,     0,    78,   111,     0,
      48,     0,    47,     0,     0,     0,     0,    53,    51,     0,
       0,     0,    60,   112,    45
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    19,    20,    21,   164,    22,    75,    56,    57,   106,
      91,   152,   153,   154,   165,   166,   179,   180,    92,   100,
     101,    94,   102,    95,   103,    43,    24,    44,    25,    26,
      36,    81,   175,   188
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -59
static const yytype_int16 yypact[] =
{
     114,   204,   204,   204,   -20,   -52,   -32,   -29,   -59,   -28,
     -17,    -9,   -59,   -59,   -59,   204,   204,   204,   204,    12,
       6,   -59,   516,   -59,   419,   -59,   -49,    24,    24,    24,
     -59,   -59,    -6,   204,   204,   204,   158,   204,   204,    15,
      24,   -59,    -5,   -59,   -46,   -59,   -59,   204,   204,   204,
      15,    15,    73,   204,   204,    15,   -59,   516,   204,   204,
     204,   204,   204,   204,   204,   204,   204,   204,   204,   204,
     204,   204,    48,    14,    22,    27,   289,   315,   341,   -59,
      29,   -59,    61,   234,    43,   -59,   204,   -59,    26,    34,
      45,   -59,    15,    42,    83,   -59,    15,   -59,   -59,   -59,
     -59,    73,    63,    73,    59,    64,    65,    71,   -59,   -59,
     419,   203,   436,   452,   467,    24,   -25,   -25,   -25,   -25,
     -25,   -25,   -59,   -59,   -59,    67,    62,    15,   -59,   -59,
     -59,   204,   204,   204,    75,   -59,   -59,   -59,   -59,   -59,
     -12,   204,   -59,   -59,    15,   -59,   -59,   -59,   -59,    97,
     -59,   -59,   -59,    77,    84,   -59,    82,   367,   393,   -59,
     113,   -59,   132,   119,    92,    91,   -59,    93,    96,   -59,
      15,   -59,   -59,   -59,   -59,   -26,   136,   123,   -59,   -39,
     -59,   204,   -59,   -59,   133,   -59,    95,   -59,   -59,   130,
     -59,   119,   -59,   -36,   204,   204,   149,   -59,   -59,   106,
     262,   -12,   -59,   -59,   -59
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
     -59,   -59,   -59,   150,   165,   -59,   -59,   115,   -59,   -59,
      79,     4,   -59,   -59,   -18,   -59,   -59,   -16,   -59,   -58,
     -59,    44,   -59,    99,   -47,    35,    -1,    10,   -11,   -59,
     -59,   -59,   -59,   -59
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -11
static const yytype_int16 yytable[] =
{
      27,    28,    29,   160,    96,    58,   161,    33,   186,    72,
      73,    30,    45,    31,    40,    41,    86,    74,    87,    30,
       4,    31,   162,   191,   198,   192,    86,    34,    84,    71,
      35,    37,    76,    77,    78,    23,    82,    83,   187,    93,
      97,    97,    38,   143,   107,   145,    12,    13,   125,    96,
      39,   163,    42,   -10,    58,    85,   126,   109,   110,   111,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,    80,    65,    66,    67,    68,    69,    70,    71,   123,
     124,    93,    88,    89,    90,    97,   127,   136,   104,   105,
      97,    58,    97,    98,   131,   137,    59,    60,    61,    62,
      63,    64,    99,   134,    12,    13,   138,   140,   141,    65,
      66,    67,    68,    69,    70,    71,   155,     1,     2,     3,
     146,   135,   144,   132,   149,   147,   148,   150,     4,   151,
     169,   157,   158,    97,     5,     6,     7,     8,   159,   170,
       9,    10,    11,   172,   171,    12,    13,   176,    14,   177,
     178,   181,   182,   189,   183,    15,   184,   190,   194,   155,
     195,     1,     2,     3,   196,   201,   156,   202,    16,    32,
      46,   139,   108,    17,   185,   197,   167,    18,     5,     6,
       7,     8,    79,   204,     9,    10,    11,     0,   168,    12,
      13,   193,    14,     0,   200,   142,     0,     0,     0,    15,
       0,     0,     0,     0,     0,     0,     0,     1,     2,     3,
       0,     0,    16,     0,     0,     0,     0,    17,     0,     0,
       0,    18,     0,     0,     5,     6,     7,     8,     0,   199,
       9,    10,    11,    58,     0,    12,    13,     0,    14,     0,
      61,    62,    63,    64,     0,    15,     0,     0,     0,     0,
       0,    65,    66,    67,    68,    69,    70,    71,    16,     0,
       0,     0,     0,    17,    58,     0,     0,    18,     0,    59,
      60,    61,    62,    63,    64,     0,     0,     0,     0,     0,
       0,     0,    65,    66,    67,    68,    69,    70,    71,     0,
       0,     0,    58,     0,     0,     0,   133,    59,    60,    61,
      62,    63,    64,     0,     0,     0,     0,     0,     0,     0,
      65,    66,    67,    68,    69,    70,    71,     0,     0,    58,
       0,     0,     0,   203,    59,    60,    61,    62,    63,    64,
       0,     0,     0,     0,     0,     0,     0,    65,    66,    67,
      68,    69,    70,    71,     0,    58,     0,     0,     0,   128,
      59,    60,    61,    62,    63,    64,     0,     0,     0,     0,
       0,     0,     0,    65,    66,    67,    68,    69,    70,    71,
       0,    58,     0,     0,     0,   129,    59,    60,    61,    62,
      63,    64,     0,     0,     0,     0,     0,     0,     0,    65,
      66,    67,    68,    69,    70,    71,     0,    58,     0,     0,
       0,   130,    59,    60,    61,    62,    63,    64,     0,     0,
       0,     0,     0,     0,     0,    65,    66,    67,    68,    69,
      70,    71,     0,    58,     0,     0,     0,   173,    59,    60,
      61,    62,    63,    64,     0,     0,     0,     0,     0,     0,
       0,    65,    66,    67,    68,    69,    70,    71,     0,    58,
       0,     0,     0,   174,    59,    60,    61,    62,    63,    64,
       0,     0,     0,     0,     0,     0,    58,    65,    66,    67,
      68,    69,    70,    71,    62,    63,    64,     0,     0,     0,
       0,     0,    58,     0,    65,    66,    67,    68,    69,    70,
      71,    63,    64,     0,     0,     0,     0,    58,     0,     0,
      65,    66,    67,    68,    69,    70,    71,    64,     0,     0,
       0,     0,     0,     0,     0,    65,    66,    67,    68,    69,
      70,    71,    47,    48,    49,    50,    51,    52,    53,    54,
       0,     0,     0,     0,     0,    55
};

static const yytype_int16 yycheck[] =
{
       1,     2,     3,    15,    51,    30,    18,    59,    34,    58,
      59,    31,     0,    33,    15,    16,    62,    66,    64,    31,
      14,    33,    34,    62,    60,    64,    62,    59,    39,    54,
      59,    59,    33,    34,    35,     0,    37,    38,    64,    50,
      51,    52,    59,   101,    55,   103,    31,    32,    34,    96,
      59,    63,    17,    59,    30,    60,    34,    58,    59,    60,
      61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    36,    48,    49,    50,    51,    52,    53,    54,    31,
      32,    92,    47,    48,    49,    96,    59,    61,    53,    54,
     101,    30,   103,    20,    65,    61,    35,    36,    37,    38,
      39,    40,    29,    60,    31,    32,    61,    65,    25,    48,
      49,    50,    51,    52,    53,    54,   127,     3,     4,     5,
      61,    86,    59,    62,    53,    61,    61,    60,    14,    67,
      33,   132,   133,   144,    20,    21,    22,    23,    63,    62,
      26,    27,    28,    61,    60,    31,    32,    34,    34,    17,
      31,    59,    61,    17,    61,    41,    60,    34,    25,   170,
      65,     3,     4,     5,    34,    16,   131,    61,    54,     4,
      20,    92,    57,    59,   170,   191,   141,    63,    20,    21,
      22,    23,    24,   201,    26,    27,    28,    -1,   144,    31,
      32,   181,    34,    -1,   195,    96,    -1,    -1,    -1,    41,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,     4,     5,
      -1,    -1,    54,    -1,    -1,    -1,    -1,    59,    -1,    -1,
      -1,    63,    -1,    -1,    20,    21,    22,    23,    -1,   194,
      26,    27,    28,    30,    -1,    31,    32,    -1,    34,    -1,
      37,    38,    39,    40,    -1,    41,    -1,    -1,    -1,    -1,
      -1,    48,    49,    50,    51,    52,    53,    54,    54,    -1,
      -1,    -1,    -1,    59,    30,    -1,    -1,    63,    -1,    35,
      36,    37,    38,    39,    40,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    48,    49,    50,    51,    52,    53,    54,    -1,
      -1,    -1,    30,    -1,    -1,    -1,    62,    35,    36,    37,
      38,    39,    40,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      48,    49,    50,    51,    52,    53,    54,    -1,    -1,    30,
      -1,    -1,    -1,    61,    35,    36,    37,    38,    39,    40,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,    49,    50,
      51,    52,    53,    54,    -1,    30,    -1,    -1,    -1,    60,
      35,    36,    37,    38,    39,    40,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    48,    49,    50,    51,    52,    53,    54,
      -1,    30,    -1,    -1,    -1,    60,    35,    36,    37,    38,
      39,    40,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,
      49,    50,    51,    52,    53,    54,    -1,    30,    -1,    -1,
      -1,    60,    35,    36,    37,    38,    39,    40,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    48,    49,    50,    51,    52,
      53,    54,    -1,    30,    -1,    -1,    -1,    60,    35,    36,
      37,    38,    39,    40,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    48,    49,    50,    51,    52,    53,    54,    -1,    30,
      -1,    -1,    -1,    60,    35,    36,    37,    38,    39,    40,
      -1,    -1,    -1,    -1,    -1,    -1,    30,    48,    49,    50,
      51,    52,    53,    54,    38,    39,    40,    -1,    -1,    -1,
      -1,    -1,    30,    -1,    48,    49,    50,    51,    52,    53,
      54,    39,    40,    -1,    -1,    -1,    -1,    30,    -1,    -1,
      48,    49,    50,    51,    52,    53,    54,    40,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    48,    49,    50,    51,    52,
      53,    54,     6,     7,     8,     9,    10,    11,    12,    13,
      -1,    -1,    -1,    -1,    -1,    19
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,    14,    20,    21,    22,    23,    26,
      27,    28,    31,    32,    34,    41,    54,    59,    63,    69,
      70,    71,    73,    93,    94,    96,    97,    94,    94,    94,
      31,    33,    72,    59,    59,    59,    98,    59,    59,    59,
      94,    94,    93,    93,    95,     0,    71,     6,     7,     8,
       9,    10,    11,    12,    13,    19,    75,    76,    30,    35,
      36,    37,    38,    39,    40,    48,    49,    50,    51,    52,
      53,    54,    58,    59,    66,    74,    94,    94,    94,    24,
      93,    99,    94,    94,    96,    60,    62,    64,    93,    93,
      93,    78,    86,    96,    89,    91,    92,    96,    20,    29,
      87,    88,    90,    92,    93,    93,    77,    96,    75,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    31,    32,    34,    34,    59,    60,    60,
      60,    65,    62,    62,    60,    93,    61,    61,    61,    78,
      65,    25,    91,    87,    59,    87,    61,    61,    61,    53,
      60,    67,    79,    80,    81,    96,    93,    94,    94,    63,
      15,    18,    34,    63,    72,    82,    83,    93,    89,    33,
      62,    60,    61,    60,    60,   100,    34,    17,    31,    84,
      85,    59,    61,    61,    60,    79,    34,    64,   101,    17,
      34,    62,    64,    95,    25,    65,    34,    85,    60,    93,
      94,    16,    61,    61,    82
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 3:
#line 189 "parser.y"
    { PARSER.module->add_spec(stack((yyvsp[(1) - (1)])));
                       PARSER.module->used=true; }
    break;

  case 9:
#line 204 "parser.y"
    { new_module((yyvsp[(2) - (2)])); }
    break;

  case 10:
#line 205 "parser.y"
    { new_module((yyvsp[(2) - (2)])); }
    break;

  case 16:
#line 214 "parser.y"
    { PARSER.module->add_init(stack((yyvsp[(2) - (2)])), stack((yyvsp[(1) - (2)])).location()); }
    break;

  case 17:
#line 215 "parser.y"
    { PARSER.module->add_init(stack((yyvsp[(2) - (3)])), stack((yyvsp[(1) - (3)])).location()); }
    break;

  case 19:
#line 217 "parser.y"
    { PARSER.module->add_trans(stack((yyvsp[(2) - (2)])), stack((yyvsp[(1) - (2)])).location()); }
    break;

  case 20:
#line 218 "parser.y"
    { PARSER.module->add_trans(stack((yyvsp[(2) - (3)])), stack((yyvsp[(1) - (3)])).location()); }
    break;

  case 22:
#line 220 "parser.y"
    { PARSER.module->add_spec(stack((yyvsp[(2) - (2)])), stack((yyvsp[(1) - (2)])).location()); }
    break;

  case 23:
#line 221 "parser.y"
    { PARSER.module->add_spec(stack((yyvsp[(2) - (3)])), stack((yyvsp[(1) - (3)])).location()); }
    break;

  case 29:
#line 227 "parser.y"
    { PARSER.module->add_invar(stack((yyvsp[(2) - (2)])), stack((yyvsp[(1) - (2)])).location()); }
    break;

  case 30:
#line 228 "parser.y"
    { PARSER.module->add_invar(stack((yyvsp[(2) - (3)])), stack((yyvsp[(1) - (3)])).location()); }
    break;

  case 32:
#line 230 "parser.y"
    { PARSER.module->add_fairness(stack((yyvsp[(2) - (2)])), stack((yyvsp[(1) - (2)])).location()); }
    break;

  case 33:
#line 231 "parser.y"
    { PARSER.module->add_fairness(stack((yyvsp[(2) - (3)])), stack((yyvsp[(1) - (3)])).location()); }
    break;

  case 37:
#line 238 "parser.y"
    {
  const irep_idt &identifier=stack((yyvsp[(1) - (3)])).get("identifier");
  smv_parse_treet::mc_vart &var=PARSER.module->vars[identifier];

  if(var.identifier!="")
  {
    yyerror("variable `"+id2string(identifier)+"' already declared extern");
    YYERROR;
  }
  else
    var.identifier=stack((yyvsp[(3) - (3)])).id_string();
}
    break;

  case 40:
#line 257 "parser.y"
    {
  const irep_idt &identifier=stack((yyvsp[(1) - (1)])).get("identifier");
  smv_parse_treet::mc_vart &var=PARSER.module->vars[identifier];
  var.var_class=smv_parse_treet::mc_vart::ARGUMENT;
  PARSER.module->ports.push_back(identifier);
}
    break;

  case 45:
#line 274 "parser.y"
    {
  init((yyval), "array");
  int start=atoi(stack((yyvsp[(2) - (6)])).id().c_str());
  int end=atoi(stack((yyvsp[(4) - (6)])).id().c_str());

  if(end < start)
  {
    yyerror("array must end with number >= `"+i2string(start)+"'");
    YYERROR;
  }

  stack((yyval)).set("size", end-start+1);
  stack((yyval)).set("offset", start);
  stack((yyval)).set("subtype", stack((yyvsp[(6) - (6)])));
}
    break;

  case 46:
#line 289 "parser.y"
    { init((yyval), "bool"); }
    break;

  case 47:
#line 290 "parser.y"
    { (yyval)=(yyvsp[(2) - (3)]); }
    break;

  case 48:
#line 292 "parser.y"
    {
              init((yyval), "range");
              stack((yyval)).set("from", stack((yyvsp[(1) - (3)])));
              stack((yyval)).set("to", stack((yyvsp[(3) - (3)])));
            }
    break;

  case 50:
#line 301 "parser.y"
    {
              init((yyval), "submodule");
              stack((yyval)).set("identifier",
                            smv_module_symbol(stack((yyvsp[(1) - (1)])).id_string()));
            }
    break;

  case 51:
#line 307 "parser.y"
    {
              init((yyval), "submodule");
              stack((yyval)).set("identifier",
                            smv_module_symbol(stack((yyvsp[(1) - (4)])).id_string()));
              stack((yyval)).operands().swap(stack((yyvsp[(3) - (4)])).operands());
            }
    break;

  case 52:
#line 316 "parser.y"
    {
               init((yyval), "enum");
               stack((yyval)).add("elements").get_sub().push_back(irept(stack((yyvsp[(1) - (1)])).id()));
              }
    break;

  case 53:
#line 321 "parser.y"
    {
               (yyval)=(yyvsp[(1) - (3)]);
               stack((yyval)).add("elements").get_sub().push_back(irept(stack((yyvsp[(3) - (3)])).id())); 
              }
    break;

  case 54:
#line 328 "parser.y"
    {
                (yyval)=(yyvsp[(1) - (1)]);
                PARSER.module->enum_set.insert(stack((yyvsp[(1) - (1)])).id_string());
              }
    break;

  case 55:
#line 335 "parser.y"
    {
  const irep_idt &identifier=stack((yyvsp[(1) - (4)])).get("identifier");
  smv_parse_treet::mc_vart &var=PARSER.module->vars[identifier];

  switch(var.var_class)
  {
  case smv_parse_treet::mc_vart::UNKNOWN:
    var.type=(typet &)stack((yyvsp[(3) - (4)]));
    var.var_class=smv_parse_treet::mc_vart::DECLARED;
    break;

  case smv_parse_treet::mc_vart::DEFINED:
    yyerror("variable `"+id2string(identifier)+"' already defined");
    YYERROR;
    break;

  case smv_parse_treet::mc_vart::DECLARED:
    yyerror("variable `"+id2string(identifier)+"' already declared as variable");
    YYERROR;
    break;
  
  case smv_parse_treet::mc_vart::ARGUMENT:
    yyerror("variable `"+id2string(identifier)+"' already declared as argument");
    YYERROR;
    break;
  
  default:
    assert(false);
  }
}
    break;

  case 60:
#line 374 "parser.y"
    {
  binary((yyval), (yyvsp[(3) - (7)]), "=", (yyvsp[(6) - (7)]));

  if(stack((yyvsp[(1) - (7)])).id()=="next")
  {
    exprt &op=stack((yyval)).op0();
    exprt tmp("smv_next");
    tmp.operands().resize(1);
    tmp.op0().swap(op);
    tmp.swap(op);
    PARSER.module->add_trans(stack((yyval)));
  }
  else
    PARSER.module->add_init(stack((yyval)));
}
    break;

  case 62:
#line 394 "parser.y"
    { init((yyval), "init"); }
    break;

  case 63:
#line 395 "parser.y"
    { init((yyval), "next"); }
    break;

  case 66:
#line 403 "parser.y"
    {
  const irep_idt &identifier=stack((yyvsp[(1) - (4)])).get("identifier");
  smv_parse_treet::mc_vart &var=PARSER.module->vars[identifier];

  switch(var.var_class)
  {
  case smv_parse_treet::mc_vart::UNKNOWN:
    var.type.make_nil();
    var.var_class=smv_parse_treet::mc_vart::DEFINED;
    break;

  case smv_parse_treet::mc_vart::DECLARED:
    var.var_class=smv_parse_treet::mc_vart::DEFINED;
    break;

  case smv_parse_treet::mc_vart::DEFINED:
    yyerror("variable `"+id2string(identifier)+"' already defined");
    YYERROR;
    break;
  
  case smv_parse_treet::mc_vart::ARGUMENT:
    yyerror("variable `"+id2string(identifier)+"' already declared as argument");
    YYERROR;
    break;
  
  default:
    assert(false);
  }

  binary((yyval), (yyvsp[(1) - (4)]), "=", (yyvsp[(3) - (4)]));
  PARSER.module->add_define(stack((yyval)));
}
    break;

  case 69:
#line 441 "parser.y"
    { init((yyval), "smv_next"); mto((yyval), (yyvsp[(3) - (4)])); }
    break;

  case 70:
#line 442 "parser.y"
    { (yyval)=(yyvsp[(2) - (3)]); }
    break;

  case 71:
#line 443 "parser.y"
    { (yyval)=(yyvsp[(2) - (3)]); stack((yyval)).id("smv_nondet_choice"); }
    break;

  case 72:
#line 444 "parser.y"
    { init((yyval), "inc"); mto((yyval), (yyvsp[(3) - (4)])); }
    break;

  case 73:
#line 445 "parser.y"
    { init((yyval), "dec"); mto((yyval), (yyvsp[(3) - (4)])); }
    break;

  case 74:
#line 446 "parser.y"
    { j_binary((yyval), (yyvsp[(3) - (6)]), "+", (yyvsp[(5) - (6)])); }
    break;

  case 75:
#line 447 "parser.y"
    { init((yyval), "-"); mto((yyval), (yyvsp[(3) - (6)])); mto((yyval), (yyvsp[(5) - (6)])); }
    break;

  case 76:
#line 448 "parser.y"
    { init((yyval), "number_constant"); stack((yyval)).set("value", stack((yyvsp[(1) - (1)])).id()); }
    break;

  case 77:
#line 449 "parser.y"
    { (yyval)=(yyvsp[(2) - (3)]); }
    break;

  case 78:
#line 450 "parser.y"
    { init((yyval), "switch"); mto((yyval), (yyvsp[(3) - (7)])); mto((yyval), (yyvsp[(6) - (7)])); }
    break;

  case 79:
#line 451 "parser.y"
    { init((yyval), "unary-"); mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 80:
#line 452 "parser.y"
    { j_binary((yyval), (yyvsp[(1) - (3)]), "+", (yyvsp[(3) - (3)])); }
    break;

  case 81:
#line 453 "parser.y"
    { j_binary((yyval), (yyvsp[(1) - (3)]), "-", (yyvsp[(3) - (3)])); }
    break;

  case 82:
#line 454 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "=", (yyvsp[(3) - (3)])); }
    break;

  case 83:
#line 455 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "=>", (yyvsp[(3) - (3)])); }
    break;

  case 84:
#line 456 "parser.y"
    { j_binary((yyval), (yyvsp[(1) - (3)]), "xor", (yyvsp[(3) - (3)])); }
    break;

  case 85:
#line 457 "parser.y"
    { j_binary((yyval), (yyvsp[(1) - (3)]), "or", (yyvsp[(3) - (3)])); }
    break;

  case 86:
#line 458 "parser.y"
    { j_binary((yyval), (yyvsp[(1) - (3)]), "and", (yyvsp[(3) - (3)])); }
    break;

  case 87:
#line 459 "parser.y"
    { init((yyval), "not"); mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 88:
#line 460 "parser.y"
    { init((yyval), "AX");  mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 89:
#line 461 "parser.y"
    { init((yyval), "AF");  mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 90:
#line 462 "parser.y"
    { init((yyval), "AG");  mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 91:
#line 463 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "=",  (yyvsp[(3) - (3)])); }
    break;

  case 92:
#line 464 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "notequal", (yyvsp[(3) - (3)])); }
    break;

  case 93:
#line 465 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "<",  (yyvsp[(3) - (3)])); }
    break;

  case 94:
#line 466 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "<=", (yyvsp[(3) - (3)])); }
    break;

  case 95:
#line 467 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), ">",  (yyvsp[(3) - (3)])); }
    break;

  case 96:
#line 468 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), ">=", (yyvsp[(3) - (3)])); }
    break;

  case 97:
#line 469 "parser.y"
    { binary((yyval), (yyvsp[(1) - (3)]), "smv_union", (yyvsp[(3) - (3)])); }
    break;

  case 98:
#line 472 "parser.y"
    { init((yyval)); mto((yyval), (yyvsp[(1) - (1)])); }
    break;

  case 99:
#line 473 "parser.y"
    { (yyval)=(yyvsp[(1) - (3)]); mto((yyval), (yyvsp[(3) - (3)])); }
    break;

  case 100:
#line 477 "parser.y"
    {
                 const std::string &id=stack((yyvsp[(1) - (1)])).id_string();

                 bool is_enum=(PARSER.module->enum_set.find(id)!=
                               PARSER.module->enum_set.end());
                 bool is_var=(PARSER.module->vars.find(id)!=
                              PARSER.module->vars.end());

                 if(is_var && is_enum)
                  {
                   yyerror("identifier `"+id+"' is ambiguous");
                   YYERROR;
                  }
                 else if(is_enum)
                  {
                   init((yyval), "enum_constant");
                   stack((yyval)).type()=typet("enum");
                   stack((yyval)).set("value", stack((yyvsp[(1) - (1)])).id());
                  }
                 else // not an enum, probably a variable
                  {
                   init((yyval), "symbol");
                   stack((yyval)).set("identifier", stack((yyvsp[(1) - (1)])).id());
                   //PARSER.module->vars[stack($1).id()];
                  }
                }
    break;

  case 101:
#line 506 "parser.y"
    {
                init((yyval), std::string(stack((yyvsp[(1) - (1)])).id_string(), 1)); // remove backslash
              }
    break;

  case 103:
#line 511 "parser.y"
    {
                std::string id(stack((yyvsp[(1) - (3)])).id_string());
                id+=".";
                id+=std::string(stack((yyvsp[(3) - (3)])).id_string(), 1); // remove backslash
                init((yyval), id);
              }
    break;

  case 104:
#line 518 "parser.y"
    {
                std::string id(stack((yyvsp[(1) - (3)])).id_string());
                id+=".";
                id+=stack((yyvsp[(3) - (3)])).id_string();
                init((yyval), id);
              }
    break;

  case 105:
#line 525 "parser.y"
    {
                std::string id(stack((yyvsp[(1) - (4)])).id_string());
                id+="[";
                id+=stack((yyvsp[(3) - (4)])).id_string();
                id+="]";
                init((yyval), id);
              }
    break;

  case 106:
#line 533 "parser.y"
    {
                std::string id(stack((yyvsp[(1) - (4)])).id_string());
                id+="(";
                id+=stack((yyvsp[(3) - (4)])).id_string();
                id+=")";
                init((yyval), id);
              }
    break;

  case 107:
#line 542 "parser.y"
    { init((yyval), "smv_cases"); }
    break;

  case 108:
#line 543 "parser.y"
    { (yyval)=(yyvsp[(1) - (2)]); mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 109:
#line 546 "parser.y"
    { binary((yyval), (yyvsp[(1) - (4)]), "case", (yyvsp[(3) - (4)])); }
    break;

  case 110:
#line 549 "parser.y"
    { init((yyval), "switches"); }
    break;

  case 111:
#line 550 "parser.y"
    { (yyval)=(yyvsp[(1) - (2)]); mto((yyval), (yyvsp[(2) - (2)])); }
    break;

  case 112:
#line 553 "parser.y"
    { init((yyval), "switch"); mto((yyval), (yyvsp[(1) - (4)])); mto((yyval), (yyvsp[(3) - (4)])); }
    break;


/* Line 1267 of yacc.c.  */
#line 2386 "y.tab.cpp"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 556 "parser.y"


int yysmverror(const std::string &error)
{
  PARSER.parse_error(error, yytext);
  return 0;
}

#undef yyerror


