/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

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

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     ASSERT = 258,
     NEWLINE = 259,
     STOP = 260,
     SKIP = 261,
     CHAOS = 262,
     NAME = 263,
     NUMBER = 264,
     CSPTRUE = 265,
     CSPFALSE = 266,
     CHANNEL = 267,
     DATATYPE = 268,
     NAMETYPE = 269,
     OPEN = 270,
     CLOSE = 271,
     LSUBST = 272,
     RSUBST = 273,
     LBRACE = 274,
     RBRACE = 275,
     LPBRACE = 276,
     RPBRACE = 277,
     EQUAL = 278,
     COMMA = 279,
     DOTDOT = 280,
     PIPE = 281,
     BECOMES = 282,
     UNION = 283,
     DIFF = 284,
     INTER = 285,
     HEAD = 286,
     TAIL = 287,
     ELSE = 288,
     THEN = 289,
     IF = 290,
     LDOT = 291,
     LAMBDA = 292,
     IN = 293,
     LET = 294,
     COLON = 295,
     AT = 296,
     RSQUARE = 297,
     PAR = 298,
     LSQUARE = 299,
     RCOMM = 300,
     LCOMM = 301,
     BACKSLASH = 302,
     INTL = 303,
     NDET = 304,
     BOX = 305,
     INTR = 306,
     SEMI = 307,
     GUARD = 308,
     WITHIN = 309,
     DOT = 310,
     OR = 311,
     AND = 312,
     NOT = 313,
     EQ = 314,
     NE = 315,
     LT = 316,
     LE = 317,
     GT = 318,
     GE = 319,
     MINUS = 320,
     PLUS = 321,
     MOD = 322,
     SLASH = 323,
     TIMES = 324,
     HASH = 325,
     CAT = 326,
     ARROW = 327,
     PLING = 328,
     QUERY = 329
   };
#endif
/* Tokens.  */
#define ASSERT 258
#define NEWLINE 259
#define STOP 260
#define SKIP 261
#define CHAOS 262
#define NAME 263
#define NUMBER 264
#define CSPTRUE 265
#define CSPFALSE 266
#define CHANNEL 267
#define DATATYPE 268
#define NAMETYPE 269
#define OPEN 270
#define CLOSE 271
#define LSUBST 272
#define RSUBST 273
#define LBRACE 274
#define RBRACE 275
#define LPBRACE 276
#define RPBRACE 277
#define EQUAL 278
#define COMMA 279
#define DOTDOT 280
#define PIPE 281
#define BECOMES 282
#define UNION 283
#define DIFF 284
#define INTER 285
#define HEAD 286
#define TAIL 287
#define ELSE 288
#define THEN 289
#define IF 290
#define LDOT 291
#define LAMBDA 292
#define IN 293
#define LET 294
#define COLON 295
#define AT 296
#define RSQUARE 297
#define PAR 298
#define LSQUARE 299
#define RCOMM 300
#define LCOMM 301
#define BACKSLASH 302
#define INTL 303
#define NDET 304
#define BOX 305
#define INTR 306
#define SEMI 307
#define GUARD 308
#define WITHIN 309
#define DOT 310
#define OR 311
#define AND 312
#define NOT 313
#define EQ 314
#define NE 315
#define LT 316
#define LE 317
#define GT 318
#define GE 319
#define MINUS 320
#define PLUS 321
#define MOD 322
#define SLASH 323
#define TIMES 324
#define HASH 325
#define CAT 326
#define ARROW 327
#define PLING 328
#define QUERY 329




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yycvclval;

