### ESBMC Architecture and Verification Process

(The Efficient SMT-based Context-Bounded Model Checker (ESBMC) is a software model checker designed to automatically detect bugs or prove their absence in programs written across multiple programming languages. The figure below illustrates ESBMC's verification architecture.

<img width="2004" height="1146" alt="image" src="https://github.com/user-attachments/assets/a98df385-193c-4717-a58b-9ae6cdc1e14f" />

Note that the verification process begins when ESBMC accepts source code written in any of its supported languages: C/C++, CUDA, CHERI-C, Java, Kotlin, Solidity, Python, or Rust. This multi-language support makes ESBMC valuable for software development environments where applications often integrate components written in different programming languages.

Upon receiving the input program, ESBMC first parses the source code and constructs an Abstract Syntax Tree (AST) that represents the program's structure and semantics. This AST is then systematically transformed into an intermediate representation called a GOTO program. This state transition system captures all possible execution paths through the original code while normalizing control flow constructs.

The core of ESBMC's analysis lies in its symbolic execution engine, which processes the GOTO program by systematically exploring execution paths. During this phase, the engine unrolls loops and recursive calls (up to specified bounds) and generates a sequence of Static Single Assignment (SSA) statements. In SSA form, each variable is assigned exactly once, which simplifies analysis and enables more precise reasoning about program behavior. ESBMC also implements a state-of-the-art proof by induction algorithm, which allows its application to unbounded verification.

The final transformation converts these SSA statements into a Satisfiability Modulo Theories (SMT) formula, which is a fragment of a first-order logical expression that encodes the program's execution constraints and the negation of desired properties. This formula has the following property: it is satisfiable if and only if there exists an execution path that violates the specified properties, indicating the presence of bugs or errors in the original program. When the SMT solver finds the formula unsatisfiable, this proves that no such error-inducing execution path exists within the specified bounds.

This architecture enables ESBMC to detect a wide range of software defects, including buffer overflows, arithmetic overflow, null pointer dereferences, assertion violations, and violations of user-specified properties, making it an essential tool for safety-critical software development.
