# The ESBMC model checker

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/d14d06e975644907a2eb9521e09ccfe4)](https://app.codacy.com/gh/esbmc/esbmc?utm_source=github.com&utm_medium=referral&utm_content=esbmc/esbmc&utm_campaign=Badge_Grade_Dashboard)
[![GitHub All Releases](https://img.shields.io/github/downloads/esbmc/esbmc/total.svg)](https://github.com/esbmc/esbmc/releases)

ESBMC (the Efficient SMT-based Context-Bounded Model Checker) is a mature, permissively licensed open-source context-bounded model checker that automatically detects or proves the absence of runtime errors in single- and multi-threaded C, C++, CUDA, CHERI, Kotlin, Python, Rust, and Solidity programs. It can automatically verify predefined safety properties (e.g., bounds check, pointer safety, overflow) and user-defined program assertions. 

ESBMC supports: 

- The Clang compiler as its C/C++/CHERI/CUDA frontend;
- The Soot framework via Jimple as its Java/Kotlin frontend;
- The CPython 3.10 parser as its [Python frontend](./src/python-frontend/README.md); the first SMT-based bounded model checker for Python programs;
- Implements the Solidity grammar production rules as its Solidity frontend;
- Supports IEEE floating-point arithmetic for various SMT solvers.

ESBMC implements state-of-the-art incremental BMC and *k*-induction proof-rule algorithms based on Satisfiability Modulo Theories (SMT) and Constraint Programming (CP) solvers.

We provide some background material/publications to help you understand what ESBMC can offer. These are available [online](https://ssvlab.github.io/esbmc/publications.html). For further information about our main components, check the ESBMC [architecture](https://github.com/esbmc/esbmc/blob/master/ARCHITECTURE.md).

Our main website is [esbmc.org](http://esbmc.org). 

### Features

ESBMC detects errors in software by simulating a finite prefix of the program execution with all possible inputs. Classes of implementation errors that can be detected include:
 * User-specified assertion failures
 * Out-of-bounds array access
 * Illegal pointer dereferences, such as:
   * Dereferencing null
   * Performing an out-of-bounds dereference
   * Double-free of malloc'd memory
   * Misaligned memory access
 * Integer overflows
 * Undefined behavior on shift operations
 * Floating-point for NaN
 * Divide by zero
 * Memory leaks

Concurrent software (using the pthread API) is verified by explicitly exploring interleavings, producing one symbolic execution per interleaving. By default, pointer-safety, array-out-of-bounds, division-by-zero, and user-specified assertions will be checked for; one can also specify options to check concurrent programs for:
 * Deadlock (only on pthread mutexes and conditional variables)
 * Data races (i.e., competing writes)
 * Atomicity violations at visible assignments
 * Lock acquisition ordering

By default, ESBMC performs a "lazy" depth-first search of interleavings -- it can also encode (explicitly) all interleavings into a single SMT formula.

Many SMT solvers are currently supported:
 * Z3 4.13+
 * Bitwuzla
 * Boolector 3.0+
 * MathSAT
 * CVC4
 * CVC5
 * Yices 2.2+

In addition, ESBMC can be configured to use the SMTLIB interactive text format with a pipe to communicate with an arbitrary solver process, although there are not insignificant overheads involved.

## Installing ESBMC

### Ubuntu

The easiest way to install ESBMC on Ubuntu is through our official [PPA](https://launchpad.net/~esbmc/+archive/ubuntu/esbmc), which provides releases for automatic installation:

```
sudo add-apt-repository ppa:esbmc/esbmc
sudo apt update
sudo apt install esbmc
```

This method is recommended for general users and supports Ubuntu 22.04 (Jammy) and 24.04 (Noble).

### GitHub Release

You can also download the latest ESBMC binary for Ubuntu and Windows from the [releases page](https://github.com/esbmc/esbmc/releases).

## Building ESBMC

See the [building instructions](https://esbmc.github.io/docs/development/building) document.

## How to use ESBMC

### Verifying C Programs

As an illustrative example to show some of the ESBMC features, consider the following C code:

```C
#include <stdlib.h>
int *a, *b;
int n;
#define BLOCK_SIZE 128
void foo () {
  int i;
  for (i = 0; i < n; i++)
    a[i] = -1;
  for (i = 0; i < BLOCK_SIZE - 1; i++)
    b[i] = -1;
}
int main () {
  n = BLOCK_SIZE;
  a = malloc (n * sizeof(*a));
  b = malloc (n * sizeof(*b));
  *b++ = 0;
  foo ();
  if (b[-1])
  { free(a); free(b); }
  else
  { free(a); free(b); }
  return 0;
}
```

Here, ESBMC is invoked as follows:

```
$esbmc file.c --incremental-bmc
```

Where `file.c` is the C program to be checked, and --incremental-bmc selects the incremental BMC strategy. The user can choose the SMT solver, property, and verification strategy. Note that you need `math.h` installed on your system, especially if you are running a release version; build-essential typically includes `math.h`.

For this particular C program, ESBMC provides the following output as the verification result:

```
[Counterexample]


State 1 file memory.c line 14 column 3 function main thread 0
----------------------------------------------------
  a = (signed int *)(&dynamic_1_array[0])

State 2 file memory.c line 15 column 3 function main thread 0
----------------------------------------------------
  b = (signed int *)0

State 3 file memory.c line 16 column 3 function main thread 0
----------------------------------------------------
Violated property:
  file memory.c line 16 column 3 function main
  dereference failure: NULL pointer


VERIFICATION FAILED

Bug found (k = 1)
```

We refer the user to our [documentation webpage](https://ssvlab.github.io/esbmc/documentation.html) for further examples of the ESBMC's features.

### Verifying Python Programs

ESBMC-Python supports verifying Python code with type annotations, detecting errors such as division by zero, indexing errors, arithmetic overflow, and user-defined assertions.

Example Python program to verify:

```python
import random as rand

def div1(cond: int, x: int) -> int:
    if (not cond):
        return 42 // x
    else:
       return x // 10

cond:int = rand.random()
x:int = rand.random()

assert div1(cond, x) != 1
```

**Command:**

```bash
$ esbmc main.py
```

**ESBMC Output:**

```
[Counterexample]


State 1 file main.py line 12 column 8 function random thread 0
----------------------------------------------------
  value = 2.619487e-10 (00111101 11110010 00000000 01000000 00000010 00000000 00010000 00001000)

State 3 file main.py line 12 column 8 function random thread 0
----------------------------------------------------
  value = 3.454678e-77 (00110000 00010000 00000000 01000000 00000010 00000000 00010000 00000000)

State 5 file main.py line 5 column 8 function div1 thread 0
----------------------------------------------------
Violated property:
  file main.py line 5 column 8 function div1
  division by zero
  x != 0


VERIFICATION FAILED
```

ESBMC-Python will parse the Python code, generate an Abstract Syntax Tree (AST), perform type inference, and translate it into the GOTO intermediate representation for symbolic execution and verification.
For detailed information about Python support, please take a look at the [Python Frontend Documentation](https://claude.ai/chat/src/python-frontend/README.md).

## Tutorials

We provide a short video that explains ESBMC: 

https://www.youtube.com/watch?v=uJ5Jn0sxm08&t=2182s

In a workshop between ARM Research and the University of Manchester, this video was delivered as part of a technical talk on exploiting the SAT revolution for automated software verification.

We offer a post-graduate course in software security that explains the internals of ESBMC. 

https://ssvlab.github.io/lucasccordeiro/courses/2020/01/software-security/index.html

This course unit introduces students to basic and advanced approaches to formally building verified trustworthy software systems, where trustworthiness comprises five attributes: *reliability*, *availability*, *safety*, *resilience*, and *security*.

## Selected Publications

* Charalambous, Y., Tihanyi, N., Jain, R., Sun, Y., Ferrag, M., Cordeiro, L.: [A New Era in Software Security: Towards Self-Healing Software via Large Language Models and Formal Verification](https://arxiv.org/pdf/2305.14752.pdf). 6th ACM/IEEE International Conference on Automation of Software Test (AST), 2025. [DOI](https://doi.org/10.48550/arXiv.2305.14752)

* Wu, T., Xiong, S., Manino, E., Stockwell, G., Cordeiro, L.: [Verifying Components of Arm® Confidential Computing Architecture with ESBMC](https://ssvlab.github.io/lucasccordeiro/papers/sas2024.pdf). In 31st International Symposium on Static Analysis (SAS), pp. 451-462, 2024. [DOI](https://link.springer.com/chapter/10.1007/978-3-031-74776-2_18)
 
* Farias, B., Menezes, R., de Lima Filho, E., Sun, Y., Cordeiro, L.: [ESBMC-Python: A Bounded Model Checker for Python Programs](https://doi.org/10.1145/3650212.3685304). In ISSTA 2024: Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA), pp. 1836-184, 2024. [DOI](https://doi.org/10.1145/3650212.3685304) [Presentation](https://ssvlab.github.io/lucasccordeiro/talks/issta2024_slides_1.pdf)

* Pirzada, M., Bhayat, A., Cordeiro, L., Reger, G. [LLM-Generated Invariants for Bounded Model Checking Without Loop Unrolling](https://doi.org/10.1145/3691620.3695512). In 39th IEEE/ACM International Conference on Automated Software Engineering (ASE), pp. 1395-1407, 2024. [DOI](https://doi.org/10.1145/3691620.3695512) [Presentation](https://ssvlab.github.io/lucasccordeiro/talks/ase2024_slides.pdf)
  
* Rafael Menezes, Daniel Moura, Helena Cavalcante, Rosiane de Freitas, Lucas C. Cordeiro . [ESBMC-Jimple: verifying Kotlin programs via jimple intermediate representation](https://dl.acm.org/doi/abs/10.1145/3533767.3543294) In ISSTA'22, pp. 777-780, 2022. [DOI](https://doi.org/10.1145/3533767.3543294)

* Franz Brauße, Fedor Shmarov, Rafael Menezes, Mikhail R. Gadelha, Konstantin Korovin, Giles Reger, Lucas C. Cordeiro. [ESBMC-CHERI: towards verification of C programs for CHERI platforms with ESBMC](https://dl.acm.org/doi/abs/10.1145/3533767.3543289) In ISSTA'22, pp. 773-776, 2022. [DOI](https://doi.org/10.1145/3533767.3543289) 

* Felipe R. Monteiro, Mikhail R. Gadelha, Lucas C. Cordeiro. [Model checking C++ programs.](https://onlinelibrary.wiley.com/doi/epdf/10.1002/stvr.1793) In Softw. Test. Verification Reliab. 32(1), 2022. [DOI](https://doi.org/10.1002/stvr.1793), [Video](https://www.youtube.com/watch?v=cX31c976tjM), **Open access**.

* Mikhail R. Gadelha, Lucas C. Cordeiro, Denis A. Nicole. [An Efficient Floating-Point Bit-Blasting API for Verifying C Programs.](https://ssvlab.github.io/lucasccordeiro/papers/nsv2020.pdf) In VSTTE, pp. 178-195, 2020. [DOI](https://doi.org/10.1007/978-3-030-63618-0_11)

* Mikhail Y. R. Gadelha, Felipe R. Monteiro, Jeremy Morse, Lucas C. Cordeiro, Bernd Fischer, Denis A. Nicole. [ESBMC 5.0: an industrial-strength C model checker.](https://ssvlab.github.io/lucasccordeiro/papers/ase2018.pdf) In ASE, pp. 888-891, 2018. [DOI](https://doi.org/10.1145/3238147.3240481) 

* Jeremy Morse, Lucas C. Cordeiro, Denis A. Nicole, Bernd Fischer. [Model checking LTL properties over ANSI-C programs with bounded traces.](https://ssvlab.github.io/lucasccordeiro/papers/sosym2013.pdf) In Softw. Syst. Model. 14(1), pp. 65-81, 2015. [DOI](https://doi.org/10.1007/s10270-013-0366-0)

* Mikhail Y. R. Gadelha, Hussama Ibrahim Ismail, Lucas C. Cordeiro. [Handling loops in bounded model checking of C programs via k-induction.](https://ssvlab.github.io/lucasccordeiro/papers/sttt2017.pdf) In Int. J. Softw. Tools Technol. Transf. 19(1), pp. 97-114, 2017. [DOI](https://doi.org/10.1007/s10009-015-0407-9)

* Phillipe A. Pereira, Higo F. Albuquerque, Isabela da Silva, Hendrio Marques, Felipe R. Monteiro, Ricardo Ferreira, Lucas C. Cordeiro. [SMT-based context-bounded model checking for CUDA programs.](https://ssvlab.github.io/lucasccordeiro/papers/cppe2017.pdf) In Concurr. Comput. Pract. Exp. 29(22), 2017. [DOI](https://doi.org/10.1002/cpe.3934)

* Lucas C. Cordeiro, Bernd Fischer, João Marques-Silva. [SMT-Based Bounded Model Checking for Embedded ANSI-C Software.](https://ssvlab.github.io/lucasccordeiro/papers/tse2012.pdf) In IEEE Trans. Software Eng. 38(4), pp. 957-974, 2012. [DOI](https://doi.org/10.1109/TSE.2011.59)

* Lucas C. Cordeiro, Bernd Fischer. [Verifying multi-threaded software using smt-based context-bounded model checking.](https://ssvlab.github.io/lucasccordeiro/papers/icse2011.pdf) In ICSE, pp. 331-340, 2011. [DOI](https://doi.org/10.1145/1985793.1985839)

## Awards

* Distinguished Paper Award at ASE'24
* 35 awards from international competitions on software verification (SV-COMP) and testing (Test-Comp) 2012-2024 at TACAS/FASE (Strength: Bug Finding and Code Coverage).
* Most Influential Paper Award at ASE’23
* Best Tool Paper Award at SBSeg'23
* Best Paper Award at SBESC’15
* Distinguished Paper Award at ICSE’11
   
## ESBMC-CHERI Video & Download

This [video](https://youtu.be/CsWHnmU4UMs) describes how to obtain, build, and run ESBMC-CHERI on an example.

A pre-compiled binary for Linux is available in the pre-release
[ESBMC-CHERI](https://github.com/esbmc/esbmc/releases/tag/v6.9-cheri), for other
systems/archs the [BUILDING.md](https://github.com/esbmc/esbmc/blob/cheri-clang/BUILDING.md)
document explains the necessary installation steps.

## Open source

ESBMC is open-source software mainly distributed under the Apache License 2.0. It contains a significant amount of other people's software. However, please take a look at the COPYING file to explain who owns what and under what terms it is distributed.

We'd be extremely happy to receive contributions to improve ESBMC (under the terms of the Apache License 2.0). Please file a pull request against the public GitHub repo if you'd like to submit anything. General discussion and release announcements will be made via GitHub. Please post an issue on GitHub and contact us about research or collaboration.

Please review the [developer documentation](https://esbmc.github.io/blob/master/CONTRIBUTIONS.md) if you want to contribute to ESBMC.

## Claude Code Plugin

A [Claude Code](https://docs.anthropic.com/claude-code) plugin for ESBMC is available at [esbmc/agent-marketplace](https://github.com/esbmc/agent-marketplace). It provides `/verify` and `/audit` commands, a verification skill, reference documentation, and examples for using ESBMC within Claude Code.

## Differences from CBMC

ESBMC is a fork of CBMC v2.9 (2008), the C Bounded Model Checker. The primary differences between the two are described [here](https://esbmc.github.io#cbmc-differences).

## Acknowledgments

ESBMC is a joint project of the Federal University of Amazonas (Brazil), the University of Manchester (UK), the University of Southampton (UK), and the University of Stellenbosch (South Africa).

The ESBMC development was supported by various research funding agencies, including CNPq (Brazil), CAPES (Brazil), FAPEAM (Brazil), EPSRC (UK), Royal Society (UK), British Council (UK), UKRI (UK), European Commission (Horizon 2020), foundations including Lattice, Rust, Ethereum, and companies including ARM, Intel, Motorola Mobility, Nokia Institute of Technology, Samsung, and [Veribee](https://www.veribee.co/). The ESBMC development is currently funded by ARM, EPSRC grants [EP/T026995/1](https://enncore.github.io) and [EP/V000497/1](https://scorch-project.github.io), [Ethereum Foundation](https://blog.ethereum.org/2022/07/29/academic-grants-grantee-announce), [Rust Foundation](https://rustfoundation.org/), [EU H2020 ELEGANT 957286](https://www.elegant-h2020.eu), Brazilian agencies CNPq (407885/2023-4), FAPEAM (01.02.016301.00292/2025-80), and CAPES (Finance Code 001), [Soteria project](https://soteriaresearch.org) awarded by the UK Research and Innovation for the Digital Security by Design (DSbD) Programme, and XC5 Hong Kong Limited.
