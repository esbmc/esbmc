* CBMC focuses on SAT-based encodings of unrolled programs, while ESBMC targets SMT-based encodings.
* CBMC's concurrency support is an entirely symbolic encoding of a concurrent program in one SAT formula, while ESBMC explores each interleaving individually using context-bounded verification.
* CBMC uses a modified C parser written by James Roskind and a C++ parser based on OpenC++, while ESBMC relies on the Clang front-end.
* ESBMC implements the Solidity and Python grammar production rules as its Solidity/Python frontend, while CBMC does not handle Solidity and Python programs.
* ESBMC verifies Kotlin programs with a model of the standard Kotlin libraries and checks a set of safety properties, while CBMC cannot handle Kotlin programs.
* CBMC implements k-induction, requiring three different calls: to generate the CFG, to annotate the program, and to verify it, whereas ESBMC handles the whole process in a single call. Additionally, CBMC does not have a forward condition to check if all states were reached and relies on a limited loop unwinding.
* ESBMC adds some additional types to the program's internal representation.
