### ESBMC Architecture and Verification Process

The Efficient SMT-based Context-Bounded Model Checker (ESBMC) is a tool that helps find bugs or confirm their absence in programs written in languages such as C, C++, Java, Kotlin, Python, and Solidity. The figure below shows how ESBMC's verification process works.

<img width="2029" height="1146" alt="image" src="https://github.com/user-attachments/assets/5e9be314-709d-4988-add7-0d8847e12c41" />

The verification process starts when ESBMC receives source code in any of its supported languages, such as C/C++, CUDA, CHERI-C, Java, Kotlin, Solidity, Python, or Rust (using the GOTO-Transcoder). Because it supports many languages, ESBMC works well in development environments where different programming languages are used together.

After getting the input program, ESBMC parses the source code and builds an Abstract Syntax Tree (AST) to show the program's structure and meaning. Next, it turns this AST into a GOTO program, which includes instructions such as assignment, conditional and non-conditional GOTO, skip, assert, and assume. This system captures all possible ways the code can run and makes the control flow easier to analyze.

The main part of ESBMC's analysis is its symbolic execution engine, which explores different ways the GOTO program can run. In this step, the engine expands loops and recursive calls up to set limits and creates a series of Static Single Assignment (SSA) statements. In SSA form, each variable gets assigned only once, which makes analysis easier and more accurate. ESBMC also uses a state-of-the-art proof-by-induction algorithm for unbounded verification. For more details, see ESBMC's papers at https://dblp.org/pid/42/4311.html.

In the last step, ESBMC turns these SSA statements into a Satisfiability Modulo Theories (SMT) formula. This formula uses first-order logic to describe how the program runs and what should not happen. If the formula can be satisfied, it means there is a way the program could break the rules, showing a bug or error. If the SMT solver cannot satisfy the formula, it proves there are no such errors within the given limits.

This setup allows ESBMC to find many types of bugs, such as buffer overflows, arithmetic overflows, null pointer errors, assertion failures, and problems with user-defined rules. This makes ESBMC a valuable tool for checking safety-critical software.
