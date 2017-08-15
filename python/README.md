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

The ESBMC python bindings allow access to the internal APIs and data structures
of ESBMC, without having to write any native code or depend on headers. The
entire system is contained in a single shared object. A replication of the ESBMC
command line program can be achieved in roughly thirty lines of python. Most
importantly, python code can be hooked into ESBMC at various stages, allowing 
inspection and extension. We have chosen python because:
 * It's a great language
 * Rapid prototyping and even heavy lifting is easy to achieve
 * There's a wealth of libraries already available, easily interoperability
   between BMC and other technology
 * The selected binding mechanism (Boost.Python) is (largely) type safe and
   directly encapsulates our existing C++ facilities

It's important to understand that these python bindings are just that: an API
for accessing the underlying C++ API. Thus any illegal behaviour with the API
such as an out-of-range vector access will stilly likely cause a crash. The
benefits comes from being able to compose objects / algorithms / data structures
from python with ESBMC objects.

Accordingly, these bindings come with some hard truths: using them is not a
substitute for knowing about (much of) ESBMCs internals. They don't provide
anything more than what ESBMC already does. They are no silver bullet for making
an analysis or extension work (they just make it easier). Finally, by it's
nature python is slower than C++, thus these bindings will not be as performant
as a C++ implementation.

## Building

To build these bindings you need to have your distributions python 3 development
package installed, which the configure script should automatically discover. You
also need Boost.Python to be installed, it's most likely in your distributions
libboost package.

The configure script is hard coded to build for python 3: technically the
autoconf file can be edited to build against python 2, and Boost.Python will
work just as well with that. At the time of writing, python 2 only has three
years of life left, new projects should avoid using it.

Two switches to the configure script are required to build the python bindings:
 * --enable-python
 * --enable-libesbmc
The former is self explanatory, the latter causes a shared object (.so) library
of ESBMC functions to be build. If you're not interested in the ESBMC command
line too, you might want to add --disable-esbmc to the configure command line,
which will avoid building static object files, and will save you 50% of build
time.

Once built, a libesbmc.so file will exist in the esbmc/.libs directory of your
build directory. To access ESBMC from python, symlink (or copy) the shared
object into your python path as 'esbmc.so', and import it from python:

    import esbmc

We might get around to writing some examples; in the mean time there should be
some regression tests published that illustrate accessing ESBMC apis from
python.

## Caveat emptor

As mentioned above, these bindings are not a substitute for knowing about ESBMCs
internals, and there are limitations as to what can be done. There are no good
python representations of, for example, list iterators that are used extensively
throughout ESBMC instead of pointers. Some facilities work around this, for
example goto-program jump targets will accept a python object reference.

Other APIs are not exported from ESBMC to python, for example fiddly intrinsics,
everything to do with parsing and GOTO conversion, and things like state hashing.
The most likely reasons for a facility not being exported is that it's not
particularly useful as a library or interesting, or that it has a very
complicated type signature.

We cannot preserve a stable ABI or API between ESBMC releases, and as a result
all releases are breaking releases. The good news is that python insulates
against ABI changes, and exceptions should only occur if a function signature
changes (or is deleted), which should always present on calling into ESBMC from
python.

## Runtime examination and debugging

The --break-at $insnnum command line option now drops into a python interactive
interpreter, allowing internal state to be frobbed during symex. Given that
python objects are trivially callable from C++, it's easy to install a call out
to python from ESBMC to perform accounting or statistics collection, following
the --break-at example.

Alternately, one can override *certain* object methods from python, and one can
install a python object within ESBMC, allowing one for example to override
symex\_step so that each step within ESBMC calls out to python. (An example of
this should be in the regression tests). This allows greater control over the
behaviour of ESBMC during symex, however:
 * It relies on methods being virtual in C++, which carries a performance cost
 * For some facilities, installing python objects into ESBMC is prohibitively
   complex

Extensions and new features can be prototyped in a similar manner, for example
by hooking symex\_step one can install custom intrinsics or re-interpret the
meaning of certain operations. The challenge is interacting correctly with the
rest of the model checker, for example if transforming the nature of an
assignment, the correct internal APIs must be called to ensure subsequent reads
will read the correct variable. The python bindings allow access to (almost)
all such APIs, and it is substantially easier to debug ESBMC in a REPL.

## Use as a library

ESBMC is not designed to be used as a library -- many global variables exist,
and just in general the codebase has never made concessions in favour of
handling the verification of different programs within the same process. This
is not easily solvable, and it's best to consider any python process as being
an ESBMC process with extra surrounding python logic. As a result, the esbmc
python module contains it's own globals, and attempting to create more than one
parseoptions object (the top level object) will lead to undefined behaviour.

The python bindings export facilities for parsing input files into GOTO
functions, creation of symex objects and running of the symex interpreter,
creation of SMT solver objects, and processing of counterexample traces. The top
level ESBMC logic (src/esbmc/bmc.cpp) is not exported: the different components
must be composed to replicate what you want it to do. An example is in
esbmc\_wrap\_solver.py.

## Caveat programmer

Boost.Python is pretty crazy, and while it's documented, ESBMC runs into many
corner cases. Happily in 95% of cases it can infer the signature of any function
and trivially wrap it to be a python function object. I would recommend emailing
the ESBMC users mailing list before attempting to export any other part of ESBMC
to python.

Python is reference counted: C++ is not (by default). This means that one risks
storing a reference to a python object in C++ that then expires. Some of these
cases are caught by Boost.Python, but not all of them can be. The solution is to
know the correct lifetime of all python objects referred to in C++, and to store
python references to them for the correct period.

Boost.Python cannot fully determine the most derived type of objects in C++
being accessed from python: downcasting is often required. The net effect of
this is that one cannot access fields of expressions until after calling
esbmc.downcast\_expr with the expr. (This is the analogue of having to downcast
exprs in ESBMC itself). Happily once downcast\_expr is called, the python object
can be treated like a normal expression.

However, if one adds additional attributes to the \_\_dict\_\_ of a python
object which refers to a C++ object, the extra attributes may not be preserved.
There is no additional storage in the C++ object, and accessing the C++ object
at a later date may create a different python object referring to the C++
object, with a completely new \_\_dict\_\_. To circumvent this, one must extend
the C++ object with a python object wrapper, which is beyond the scope of this
readme.

Const correctness falls by the wayside with most Boost.Python operations. As a
result expression are read-only from python, but take care when passing (for
example) an expression reference into a C++ method that will mutate it.
Otherwise you'll set fire to an expression that the rest of ESBMC thought was
const: call clone() first to avoid this.

There's no reason why one cannot repeatedly call in and out of C++ / python with
these bindings, however be aware that:
 * Exceptions in python will cause calls from C++ to immediately terminate with
   None, and if you don't clear the exception it'll continue if you return back
   into python
 * Exceptions in C++ will set fire to a random piece of python
 * Segfaults are still as fatal as ever

Nil expressions should evaluate to be equal to None, and you can pass None in
place of a nil expression. Just as with the rest of ESBMC, you need to compare
certain expression holders with None before operating on them. Some are python
objects that _refer_ to nil expressions, thus will not be None themselves, but
will still compare true to None. (This is the difference between 'is None' and
'== None').

## Future directions

Currently there are none: these bindings just provide access. It's conceivable
that (over a long time) ESBMC may become more library like, or some
transformations may become implemented in python, but that's all speculation.

## Even more caveats

It's best to assume that esbmc is going to crash when performing an operation,
unless you're very confident in what it does. I wouldn't recommend developing
in an interactive python instance, better to keep scripts on disk most of the
time.

Boost shared ptrs have a fantastic facility for acting as a python reference
and a C++ shared\_ptr reference at the same time, intimitely binding the
lifetime of each object. ESBMC contains a random mixture of std::shared\_ptr
and boost::shared\_ptr, so bear this in mind if you have inexplicable python
references hanging around.

I subclassed an STL container to better enable python access to it. I am not
proud of this.

Currently there's no way of accessing the 'old' irep from python, which is
deliberate. If it's ever supported, it'll be as opaque handles, with
migration facilities to/from irep2.

It's conceptually desirable for various enumerations to be extended, for example
to allow the creation of new SSA step types or GOTO program insn types. However,
so much of ESBMC switches on those enums that this seems infeasible as of now.
