# ESBMC python bindings

Contents:
 * Introduction
 * Building
 * Caveat emptor
 * Runtime examination and debugging
 * Use as a library
 * Caveat programmer
 * Future directions
 * Even more caveats

## Introduction

What this is
 * Bindings
 * Easier inspection of runtime stuff
 * The ability to quickly prototype logic
 * Interoperability with Other Things
 * Python is nice

What this isn't
 * Going to take away hard work
 * A substitude for knowing how ESBMC works
 * A silver bullet
 * Fast

## Building

 * Requires python-dev, of course
 * That allows for python instances within ESBMC process
 * Must be shared object to be used in any other python process
 * --enable-shared
 * Env vars
 * Boost python

## Caveat emptor

 * As mentioned, no substitude for knowledge
 * Some things aren't exported
 * Some thingis can't be exported easily
 * Python isn't going to slow down C++ development
 * Not the default

## Runtime examination and debugging

 * Can break in and have RT object in locals
 * Might be able to use this to instrument various points in ESBMC
 * Lists
 * Dicts

## Use as a library

 * ESBMC contains globals
 * One instance per process pls
 * Basic outline of the high level point that exporting stops
 * If you want to over-ride internals, you might need to implement up to that
   point
 * Might be addressed in the future via factories
 * Might also provide a high level python library that does the main driving
 * Solver API as an example?

## Caveat programmer

 * Boost.python is crazy shit
 * Reference counting
 * Downcasting, lol, and other gotchas
 * I don't really intend on making things "nice" for python, i.e. irep\_idt 
   as\_string
 * Wrapped objects?
 * Const correctness
 * XXX that compiler bug for irep2?
 * Exceptions through a C++ call stack might become fairly crazy
 * 'None' and when things are non-evalued internal references

## Future directions

 * No idea

## Even more caveats

 * Best to assume that esbmc is going to crash every time you do something
 * boost shared ptrs?
 * I subclassed an stdl container
 * old irep?
 * Probably can't extend things like enums
