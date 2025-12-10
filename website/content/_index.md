---
title: ESBMC
toc: false
---

{{< callout type="important" icon="sparkles" >}} **NEW:** Read our new website
announcement [here](/news/new-website-announcement). {{< /callout >}}


{{< cards >}}
  {{< card link="https://dblp.org/pid/42/4311" title="Publications" icon="book-open" >}}
  {{< card link="/sv-comp" title="SV-COMP" icon="book-open" >}}
  {{< card link="/test-comp" title="Test-COMP" icon="book-open" >}}
  {{< card link="https://ssvlab.github.io/" title="SSV Lab" icon="book-open" >}}
  {{< card link="https://github.com/esbmc/esbmc/releases/latest" title="GitHub" icon="github" >}}
  {{< card link="https://github.com/esbmc/esbmc-web" title="Web Interface" icon="external-link" >}}
  {{< card link="https://github.com/esbmc/vscode-esbmc" title="VSCode Extension" icon="external-link" >}}
  {{< card link="https://esbmc.github.io/esbmc-ai" title="ESBMC-AI" icon="external-link" >}}
{{< /cards >}}


{{% details title="Download" closed="true" %}}

{{< tabs items="Windows,Linux" >}}
  {{< tab >}}
    {{< cards >}}
      {{< card link="https://github.com/esbmc/esbmc/releases/latest/download/esbmc-windows.zip" title="Download" icon="download" >}}
    {{< /cards >}}
  {{< /tab >}}

  {{< tab >}}
    {{< cards >}}
      {{< card link="https://github.com/esbmc/esbmc/releases/latest/download/esbmc-linux.zip" title="Download" icon="download" >}}
    {{< /cards >}}
  {{< /tab >}}

{{< /tabs >}}

{{% /details %}}

ESBMC is a mature, permissively licensed open-source SMT-based context-bounded model checker for single and multi-threaded C, C++, CUDA, CHERI, Kotlin, Python, Rust, and Solidity programs. It automatically detects or proves the absence of runtime errors (e.g., bounds checks, pointer safety, overflow) and verifies user-defined assertions without requiring pre- or postconditions. For multi-threaded programs, ESBMC supports lazy and schedule-recording approaches, encoding verification conditions into SMT formulas solved directly by an SMT solver.

## Features

ESBMC aims to support all C features up to version 14, and detects errors in software by simulating a finite prefix of the program execution with all possible inputs. Classes of problems that can be detected include:


- The Clang compiler as its C/C++/CHERI/CUDA frontend;
- The Soot framework via Jimple as its Java/Kotlin frontend;
- The [ast](https://docs.python.org/3/library/ast.html) and [ast2json](https://pypi.org/project/ast2json/) modules as its [Python frontend](./src/python-frontend/README.md); the first SMT-based bounded model checker for Python programs;
- Implements the Solidity grammar production rules as its Solidity frontend;
- Supports IEEE floating-point arithmetic for various SMT solvers.

ESBMC implements state-of-the-art incremental BMC and *k*-induction proof-rule algorithms based on Satisfiability Modulo Theories (SMT) and Constraint Programming (CP) solvers.

We provide some background material/publications to help you understand what ESBMC can offer. These are available [online](https://ssvlab.github.io/esbmc/publications.html). For further information about our main components, check the ESBMC [architecture](https://github.com/esbmc/esbmc/blob/master/ARCHITECTURE.md).

## Applications

ESBMC has been used in a broad range of cutting-edge applications across multiple domains. If you applied ESBMC in your research, but it is not mentioned below, please, do not hesitate to contact us through our [GitHub repository](https://github.com/esbmc/esbmc).

### Recent Applications (2022-2024)

* **[ESBMC-Python: A Bounded Model Checker for Python Programs](https://dl.acm.org/doi/10.1145/3650212.3685304)** (ISSTA 2024)
>The first BMC-based Python code verifier, successfully detecting bugs in the Ethereum Consensus Specification and other Python applications.

* **[Verifying Components of Arm® Confidential Computing Architecture](https://link.springer.com/chapter/10.1007/978-3-031-74776-2_18)** (SAS 2024)
>ESBMC is used to verify the Realm Management Monitor (RMM) firmware in Arm's Confidential Computing Architecture, detecting 23 new vulnerabilities that other tools missed.

* **[LLM-Generated Invariants for Bounded Model Checking Without Loop Unrolling](https://doi.org/10.1109/ASE56229.2024.00142)** (ASE 2024)
>Integration of large language models with ESBMC to automatically generate loop invariants, eliminating the need for manual loop unrolling.

* **[ESBMC-Solidity: Verifying Blockchain Smart Contracts](https://dl.acm.org/doi/10.1145/3510454.3516855)** (ICSE 2022)
>ESBMC is used to verify Solidity smart contracts on Ethereum blockchain, detecting vulnerabilities in financial applications handling millions of dollars.

* **[ESBMC-CHERI: Hardware-Assisted Memory Safety Verification](https://dl.acm.org/doi/10.1145/3533767.3543289)** (ISSTA 2022)
>The first bounded model checker for CHERI-enabled platforms, verifying C programs with hardware-level memory protection capabilities.

* **[ESBMC-Jimple: Verifying Kotlin Programs](https://dl.acm.org/doi/10.1145/3533767.3534374)** (ISSTA 2022)
>ESBMC is extended to verify Android/Kotlin applications through the Jimple intermediate representation.

* **[Arduino Integration for Embedded Systems](https://doi.org/10.1109/SBESC60926.2023.10324055)** (SBESC 2023)
>ESBMC verification for Arduino-based embedded systems, enhancing integrity and reliability in IoT devices.

### Earlier Applications

* **[DSVerifier-Aided Verification Applied to Attitude Control Software in Unmanned Aerial Vehicles](https://ssvlab.github.io/lucasccordeiro/papers/tr2018.pdf)**
>ESBMC is used to verify embedded control software in Unmanned Aerial Vehicles.

* **[Model Checking C++ Programs](https://onlinelibrary.wiley.com/doi/10.1002/stvr.1793)** (STVR 2022)
>ESBMC provides comprehensive verification of modern C++ programs including STL containers and templates.

* **BMCLua: A Translator for Model Checking Lua Programs**
>ESBMC is used to verify a ANSI-C version of the respective Lua program.

* **[CSeq: A Sequentialization Tool for C](https://link.springer.com/chapter/10.1007%2F978-3-642-36742-7_46)**
>ESBMC is used as sequential verification back-end to model check multi-threaded programs.

* **[Sound and Unified Software Verification for Weak Memory Models](http://www.cs.ox.ac.uk/people/vincent.nimal/sas12/paper.pdf)**
>ESBMC is used as a sequential consistency software verification tool in real-life C programs.

* **[Understanding Programming Bugs in ANSI-C Software Using Bounded Model Checking Counter-Examples](https://ssvlab.github.io/lucasccordeiro/papers/ifm2012.pdf)**
>The counter-example produced by ESBMC is used to automatically debug software systems.

* **[Verifying Embedded C Software with Timing Constraints using an Untimed Model Checker](http://eprints.soton.ac.uk/272442/1/formats2011.pdf)**
>ESBMC is used as an untimed software model checker to verify real-time software.

* **[Scalable hybrid verification for embedded software](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=5763039)**
>ESBMC is used as a verification engine to model check embedded (sequential) software.

* **[Getting Rid of Store-Buffers in TSO Analysis](https://link.springer.com/chapter/10.1007%2F978-3-642-22110-1_9)**
>ESBMC is used to verify _sequential consistency_ concurrent C programs.

## Cite

If you cite ESBMC >= version 7.4, please use the [competition paper](https://arxiv.org/pdf/2312.14746.pdf) at TACAS 2024 (BibTex) as listed below:

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
