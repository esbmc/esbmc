---
title: ESBMC
toc: false
---

{{< callout type="important" icon="sparkles" >}} **NEW:** 515 commits landed on
`master` between 13 and 27 October, about three in five in the Python frontend.
Read the [development update](/news/development-update-late-october-2026).
{{< /callout >}}

{{< cards >}}
{{< card link="https://dblp.org/pid/42/4311" title="Publications" icon="book-open" >}}
{{< card link="/sv-comp" title="SV-COMP" icon="book-open" >}}
{{< card link="/test-comp" title="Test-COMP" icon="book-open" >}}
{{< card link="https://ssvlab.github.io/" title="SSV Lab" icon="book-open" >}}
{{< card link="https://github.com/esbmc/esbmc/releases/latest" title="Latest Release" icon="github" >}}
{{< card link="https://esbmc.github.io/esbmc-demo" title="ESBMC Simulator" icon="external-link" >}}
{{< card link="https://github.com/esbmc/esbmc-web" title="Web Interface" icon="external-link" >}}
{{< card link="https://github.com/esbmc/vscode-esbmc" title="VSCode Extension" icon="external-link" >}}
{{< card link="https://github.com/esbmc/agent-marketplace" title="Claude Code Plugin" icon="external-link" >}}
{{< card link="https://esbmc.github.io/esbmc-ai" title="ESBMC-AI" icon="external-link" >}}
{{< /cards >}}

ESBMC is a mature, permissively licensed open-source SMT-based context-bounded
model checker for single and multi-threaded C, C++, CUDA, CHERI-C, Java, Kotlin,
Python, and Solidity programs. It automatically detects or proves the absence of
runtime errors (e.g., bounds checks, pointer safety, overflow) and verifies
user-defined assertions without requiring pre- or postconditions. For
multi-threaded programs, ESBMC supports lazy and schedule-recording approaches,
encoding verification conditions into SMT formulas solved directly by an SMT
solver.

## Download

{{< tabs >}}

{{< tab name="Linux" >}} {{< cards >}}
{{< card link="https://github.com/esbmc/esbmc/releases/latest/download/esbmc-linux.zip" title="Download (x86_64)" icon="download" >}}
{{< card link="https://github.com/esbmc/esbmc/releases/latest/download/esbmc-linux-armv8.zip" title="Download (ARMv8)" icon="download" >}}
{{< card link="docs/setup#arm64-builds" title="ARM64 Limitations" icon="warning" >}}
{{< card link="docs/development/building" title="Build Guide" icon="cog" >}}
{{< /cards >}} {{< /tab >}}

{{< tab name="Windows" >}} {{< cards >}}
{{< card link="https://github.com/esbmc/esbmc/releases/latest/download/esbmc-windows.zip" title="Download" icon="download" >}}
{{< card link="docs/development/building" title="Build Guide" icon="cog" >}}
{{< /cards >}} {{< /tab >}}

{{< tab name="macOS" >}} {{< cards >}}
{{< card link="https://github.com/esbmc/esbmc/releases/latest/download/esbmc-macos.zip" title="Download" icon="download" >}}
{{< card link="docs/development/building" title="Build Guide" icon="cog" >}}
{{< /cards >}} {{< /tab >}}

{{< tab name="FreeBSD" >}} {{< callout type="warning" icon="warning" >}} FreeBSD
does not have any downloadable builds at the moment. You can however build from
scratch using our [build](docs/development/building) guide. {{< /callout >}}
{{< cards >}}
{{< card link="docs/development/building" title="Build Guide" icon="cog" >}}
{{< /cards >}} {{< /tab >}}

{{< /tabs >}}

## Features

ESBMC detects errors by symbolically executing a finite prefix of the program
for all possible inputs. Its frontends and encodings are:

- The Clang compiler as its C/C++/CUDA frontend, and CHERI Clang for CHERI-C
  (CHERI-C needs a separate ESBMC build; see the
  [Build Guide](docs/development/building));
- The Soot framework via Jimple as its Java/Kotlin frontend;
- The [ast](https://docs.python.org/3/library/ast.html) and
  [ast2json](https://pypi.org/project/ast2json/) modules as its
  [Python frontend](/docs/python/overview); the first SMT-based bounded model
  checker for Python programs;
- Implements the Solidity grammar production rules as its Solidity frontend;
- IEEE 754 floating-point arithmetic, encoded in the solver's floating-point
  theory or, for solvers without one, as bit-vectors.

ESBMC implements bounded model checking (BMC), incremental BMC and _k_-induction
on top of Satisfiability Modulo Theories (SMT) solvers.

We provide some background material/publications to help you understand what
ESBMC can offer. These are available
[online](https://ssvlab.github.io/esbmc/publications.html). For further
information about our main components, check the ESBMC
[architecture](https://github.com/esbmc/esbmc/blob/master/ARCHITECTURE.md).

## SV-COMP History

![ESBMC SV-COMP overall score history](./images/sv-comp-hist-combined.png)

Overall [SV-COMP](/sv-comp) score over time, by per-benchmark time budget.

## Applications

ESBMC has been used in a broad range of cutting-edge applications across
multiple domains. If you applied ESBMC in your research, but it is not mentioned
below, please, do not hesitate to contact us through our
[GitHub repository](https://github.com/esbmc/esbmc).

## CBMC Differences

ESBMC and [CBMC](https://www.cprover.org/cbmc/) share a common ancestry but have
diverged substantially. The list below reflects ESBMC 8.6 and contrasts it
with mainline CBMC.

**Frontends and language support.**

- ESBMC uses the Clang/LLVM frontend for C (up to C23), C++ (up to C++20, with
  selected C++23 features), CUDA and CHERI-C, inheriting Clang's diagnostics,
  attribute handling and template instantiation. CBMC uses its own in-tree C
  and C++ parsers.
- ESBMC ships dedicated frontends for **Python 3.10+** (the first BMC-based
  Python-code verifier), **Solidity** (smart contracts) and **Kotlin/Java** (via
  the Soot/Jimple IR). CBMC supports only C/C++ in the main tool; Java is
  handled by the separate JBMC tool, and Rust by Kani, which builds on CBMC.

**Verification backend.**

- ESBMC encodes programs directly into SMT and supports multiple solver backends
  — Z3, Bitwuzla (the default when it is built in), Boolector, CVC4/CVC5, Yices,
  MathSAT and any SMT-LIB v2 solver — using native theories for arrays,
  bit-vectors, floating-point and quantifiers. CBMC's default backend is a SAT
  solver over a bit-blasted encoding, with SMT solvers as an alternative.
- ESBMC integrates an interval abstract domain into symbolic execution. It stops
  unwinding a loop when the intervals prove the loop exits (on by default in
  plain BMC; disable with `--no-interval-symex-guard`) and can skip assertions
  the intervals prove (`--interval-symex-assert`). Both reduce the size of the
  SMT formula handed to the solver.

**Proof techniques.**

- ESBMC implements incremental BMC, _k_-induction and bidirectional
  _k_-induction in a single invocation, combining base-case, forward-condition
  and inductive-step checks in one run.
- ESBMC supports function contracts (`__ESBMC_requires`, `__ESBMC_ensures`,
  `__ESBMC_assigns`) and loop invariants (`__ESBMC_loop_invariant`), which it
  can both check and use in place of the code they describe.

**Concurrency verification.**

- ESBMC explores thread interleavings explicitly under context-bounded
  verification, with both **lazy** and **schedule-recording** exploration
  strategies and a partial-order reduction (POR) pass that prunes independent
  interleavings.
- ESBMC ships models for POSIX threads, C11 atomics and a subset of
  `std::thread` / `std::atomic`, and can verify safety, deadlock and data-race
  properties.

**Memory safety and floating-point.**

- A byte-precise memory model with object-level pointer analysis lets ESBMC
  check pointer-safety, array bounds, use-after-free, double-free and memory
  leaks across heap and stack. Floating-point follows IEEE 754, through the
  solver's floating-point theory by default or through a bit-vector encoding
  (`--fp2bv`) for solvers without one.

**Operational models and library support.**

- ESBMC ships operational models for the C standard library (compiled to GOTO
  by `c2goto`), large parts of the C++ standard library (containers, iterators,
  algorithms, strings), and Python built-ins and modules (lists, dicts,
  strings, `enum`, `math`, `re` and others, written in Python). Code that uses
  the modelled parts of these libraries can be verified without writing stubs.

**Recent competition results and industrial use.**

- ESBMC-kind placed **2nd in ReachSafety** at [SV-COMP](/sv-comp) 2025 and
  2026. At [Test-COMP](/test-comp), FuSeBMC, a test generator built on ESBMC,
  placed **1st Overall every year from 2022 to 2026**, 1st in _Cover-Error_ in
  each of those years, and 1st in _Cover-Branches_ in every year except 2025
  (2nd).
- Applications include the verification of Arm's **Realm Management
  Monitor** in the Confidential Computing Architecture (SAS 2024), Ethereum
  smart-contract auditing (ICSE 2022), Arduino-based embedded firmware (SBESC
  2023), the Ethereum Consensus Specification (ISSTA 2024) and the **ESBMC-AI**
  pipeline that pairs LLMs with the model checker to repair code.

## SPARK Ada Differences

ESBMC is sometimes compared with [SPARK](https://www.adacore.com/about-spark)
(GNATprove and the SPARK Ada subset), which targets a different region of the
verification design space. The main differences are:

**Verification methodology.**

- ESBMC is a _bounded model checker_: it symbolically executes a finite prefix
  of the program and encodes the resulting verification conditions as SMT.
  Within the bound, it finds every reachable violation of the checked
  properties. A proof for all executions requires either that the bound covers
  every execution (no unwinding assertion fails) or that _k_-induction
  converges.
- SPARK is a _deductive verifier_: GNATprove translates Ada units to Why3,
  derives Hoare-style verification conditions from user-supplied contracts, and
  discharges them with SMT solvers (Alt-Ergo, CVC5, Z3) or, for hard goals,
  interactive provers (Coq/Isabelle). It proves _unbounded_ functional
  correctness rather than searching for counterexamples.

**Supported languages.**

- ESBMC: C (up to C23), C++ (up to C++20, with selected C++23 features), CUDA,
  CHERI-C, Python, Solidity, Kotlin, Java.
- SPARK: a strict subset of Ada 2012/2022 that forbids exceptions, controlled
  types, side-effects in functions and unrestricted aliasing, with optional
  escape hatches via `SPARK_Mode => Off`.

**Annotation requirements.**

- ESBMC verifies code essentially _as-is_: built-in safety properties (pointer
  safety, overflow, bounds, leaks, data races) are checked automatically;
  user-defined assertions need only `assert` or `__ESBMC_assert`. Loop
  invariants are optional; `--interval-analysis` infers
  [interval invariants](/docs/theory/interval-analysis) and adds them as
  assumptions. When annotations _are_ supplied — `__ESBMC_requires` /
  `__ESBMC_ensures` for pre/post-conditions, `__ESBMC_assigns` for frame
  conditions, `__ESBMC_loop_invariant` for loop invariants, and
  `__ESBMC_contract` to mark functions for `--enforce-all-contracts` and
  `--replace-all-contracts` — ESBMC uses them to verify the implementation
  against its contract and to replace call sites with the contract, which
  supports compositional verification (see
  [Function Contracts](/docs/function-contracts/)).
- SPARK proofs depend on the programmer authoring `Pre`, `Post`,
  `Contract_Cases`, `Loop_Invariant`, `Loop_Variant`, `Global` and `Depends`
  aspects. Without these contracts, GNATprove can only prove Ada's built-in
  run-time checks (the "Bronze" level).

**Automation level.**

- ESBMC is push-button: one invocation reports `VERIFICATION SUCCESSFUL`,
  `VERIFICATION FAILED` (with a counterexample trace), `VERIFICATION UNKNOWN`
  (neither proved nor refuted, e.g. when _k_-induction does not converge), or
  stops at its time or memory limit.
- SPARK is mostly automatic through the _Silver_ (absence of run-time errors)
  and _Gold_ (key integrity properties) levels, given suitable contracts.
  Reaching the _Platinum_ level — full functional correctness — typically
  requires iterative contract refinement and occasionally manual proof via
  Coq/Isabelle through Why3.

**Memory and concurrency verification.**

- ESBMC reasons about heap-allocated data structures, pointer arithmetic and
  aliasing through its byte-level memory model, up to the unwinding bound, and
  verifies multi-threaded code (POSIX threads, C11 atomics, `std::thread`) by
  exploring interleavings, optionally context-bounded, with POR.
- SPARK relies on a static _ownership and borrowing_ discipline (Ada 2022) to
  rule out aliasing rather than reason about it, and restricts concurrency to
  the Ravenscar/Jorvik profiles (statically scheduled tasks, no dynamic
  allocation of tasks, bounded protected objects). This makes proofs tractable
  but excludes idiomatic concurrent C/C++ patterns.

**Target domains and typical use cases.**

- ESBMC: research, operating-system and firmware verification (e.g. Arm
  RMM/CCA), smart-contract auditing, embedded/IoT and Arduino, safety-critical
  C/C++ in industry, ML/AI safety, and education.
- SPARK: avionics (DO-178C up to Level A), defence, rail signalling (EN 50128
  SIL 4), space and security-evaluated software (Common Criteria) — domains
  where Ada is mandated and full functional proof of contracts is required by
  certification.

**Strengths and limitations.**

| Aspect      | ESBMC                                                                                                                       | SPARK Ada                                                                                                                       |
| ----------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Strengths   | Push-button, multi-language, produces counterexample traces, handles C/C++ pointers, heap and threads                       | Strong functional-correctness guarantees, mature tooling for certification, sound by construction                               |
| Limitations | Bounded by default; full proofs require _k_-induction convergence; counterexamples may be spurious if models are abstracted | Restricted language subset; effort-intensive contract authoring; weak support for dynamic aliasing and unstructured concurrency |

### Recent Applications (2022-2024)

- **[ESBMC-Python: A Bounded Model Checker for Python Programs](https://dl.acm.org/doi/10.1145/3650212.3685304)**
  (ISSTA 2024)

  > The first BMC-based Python code verifier, successfully detecting bugs in the
  > Ethereum Consensus Specification and other Python applications.

- **[Verifying Components of Arm® Confidential Computing Architecture with ESBMC](https://link.springer.com/chapter/10.1007/978-3-031-74776-2_18)**
  (SAS 2024)

  > ESBMC is used to verify the Realm Management Monitor (RMM) firmware in Arm's
  > Confidential Computing Architecture, detecting 23 new vulnerabilities that
  > other tools missed.

- **[LLM-Generated Invariants for Bounded Model Checking Without Loop Unrolling](https://doi.org/10.1145/3691620.3695512)**
  (ASE 2024)

  > Integration of large language models with ESBMC to automatically generate
  > loop invariants, eliminating the need for manual loop unrolling.

- **[ESBMC-Solidity: An SMT-Based Model Checker for Solidity Smart Contracts](https://dl.acm.org/doi/10.1145/3510454.3516855)**
  (ICSE Companion 2022)

  > ESBMC is used to verify Solidity smart contracts on Ethereum blockchain,
  > detecting vulnerabilities in financial applications handling millions of
  > dollars.

- **[ESBMC-CHERI: Hardware-Assisted Memory Safety Verification](https://dl.acm.org/doi/10.1145/3533767.3543289)**
  (ISSTA 2022)

  > The first bounded model checker for CHERI-enabled platforms, verifying C
  > programs with hardware-level memory protection capabilities.

- **[ESBMC-Jimple: Verifying Kotlin Programs](https://doi.org/10.1145/3533767.3543294)**
  (ISSTA 2022)

  > ESBMC is extended to verify Android/Kotlin applications through the Jimple
  > intermediate representation.

- **[Arduino Integration for Embedded Systems](https://doi.org/10.1109/SBESC60926.2023.10324098)**
  (SBESC 2023)
  > ESBMC verification for Arduino-based embedded systems, enhancing integrity
  > and reliability in IoT devices.

### Earlier Applications

- **[DSVerifier-Aided Verification Applied to Attitude Control Software in Unmanned Aerial Vehicles](https://ssvlab.github.io/lucasccordeiro/papers/tr2018.pdf)**

  > ESBMC is used to verify embedded control software in Unmanned Aerial
  > Vehicles.

- **[Model Checking C++ Programs](https://onlinelibrary.wiley.com/doi/10.1002/stvr.1793)**
  (STVR 2022)

  > ESBMC provides comprehensive verification of modern C++ programs including
  > STL containers and templates.

- **BMCLua: A Translator for Model Checking Lua Programs**

  > ESBMC is used to verify a ANSI-C version of the respective Lua program.

- **[CSeq: A Sequentialization Tool for C](https://link.springer.com/chapter/10.1007%2F978-3-642-36742-7_46)**

  > ESBMC is used as sequential verification back-end to model check
  > multi-threaded programs.

- **[Sound and Unified Software Verification for Weak Memory Models](http://www.cs.ox.ac.uk/people/vincent.nimal/sas12/paper.pdf)**

  > ESBMC is used as a sequential consistency software verification tool in
  > real-life C programs.

- **[Understanding Programming Bugs in ANSI-C Software Using Bounded Model Checking Counter-Examples](https://ssvlab.github.io/lucasccordeiro/papers/ifm2012.pdf)**

  > The counter-example produced by ESBMC is used to automatically debug
  > software systems.

- **[Verifying Embedded C Software with Timing Constraints using an Untimed Model Checker](http://eprints.soton.ac.uk/272442/1/formats2011.pdf)**

  > ESBMC is used as an untimed software model checker to verify real-time
  > software.

- **[Scalable hybrid verification for embedded software](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=5763039)**

  > ESBMC is used as a verification engine to model check embedded (sequential)
  > software.

- **[Getting Rid of Store-Buffers in TSO Analysis](https://link.springer.com/chapter/10.1007%2F978-3-642-22110-1_9)**
  > ESBMC is used to verify _sequential consistency_ concurrent C programs.

## Sponsors

<a href="https://cyber-reasoning.co.uk" target="_blank" rel="noopener noreferrer" title="ESBMC is sponsored by Cyber Reasoning">
  <img src="./images/sponsors/cyber-reasoning.svg" alt="Sponsored by Cyber Reasoning" style="max-width:200px;height:auto;" />
</a>

## Cite

If you cite ESBMC >= version 7.4, please use the
[competition paper](https://arxiv.org/pdf/2312.14746.pdf) at TACAS 2024 (BibTex)
as listed below:

```bib
@InProceedings{esbmc2024,
    author    = {Rafael Menezes and
                 Mohannad Aldughaim and
                 Bruno Farias and
                 Xianzhiyu Li and
                 Edoardo Manino and
                 Fedor Shmarov and
	         Kunjian Song and
	         Franz Brauße and
	         Mikhail R. Gadelha and
	         Norbert Tihanyi and
	         Konstantin Korovin and
	         Lucas C. Cordeiro},
    title     = {{ESBMC} 7.4: {H}arnessing the {P}ower of {I}ntervals},
    booktitle = {$30^{th}$ International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS'24)},
    series       = {Lecture Notes in Computer Science},
    volume       = {14572},
    pages     = {376–380},
    doi       = {https://doi.org/10.1007/978-3-031-57256-2_24},
    year      = {2024},
    publisher = {Springer}
}
```
