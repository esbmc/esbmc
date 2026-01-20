---
title: Implementing SMT Theory
---

This guide outlines the steps to integrate a new SMT theory into ESBMC. Throughout the tutorial, we will use the QF_AUFLIRA theory as an example.

## Steps to Follow

1. **Create a New Option for ESBMC**  
   Define the new option in the `options.cpp` file:  
   [Source Link](https://github.com/esbmc/esbmc/blob/master/src/esbmc/options.cpp#L258)

2. **Add New Logic to BMC**  
   Add the logic for your new SMT theory in the `bmc.cpp` file:  
   [Source Link](https://github.com/esbmc/esbmc/blob/master/src/esbmc/bmc.cpp#L159)

3. **Set the Option in Parser Options**  
   Configure the option in the parser options at:  
   [Source Link](https://github.com/esbmc/esbmc/blob/master/src/esbmc/esbmc_parseoptions.cpp#L317)

4. **Adjust Pointer Dereference Module**  
   Review and modify the pointer dereference module if necessary:  
   [Source Link](https://github.com/esbmc/esbmc/blob/master/src/pointer-analysis/dereference.cpp#L2325)

5. **Adjust SMT Backends**  
   Update the relevant SMT backends. As examples:
   - Bitwuzla: [Source Link](https://github.com/esbmc/esbmc/blob/master/src/solvers/bitwuzla/bitwuzla_conv.cpp#L28)
   - Boolector: [Source Link](https://github.com/esbmc/esbmc/blob/master/src/solvers/boolector/boolector_conv.cpp#L29)

6. **Add New Encoding Type**  
   Define the new encoding type (e.g., `real_encoding`) in the SMT converter:  
   - [Add Encoding](https://github.com/esbmc/esbmc/blob/master/src/solvers/smt/smt_conv.cpp#L69)  
   - [Set Encoding](https://github.com/esbmc/esbmc/blob/master/src/solvers/smt/smt_conv.cpp#L1908)

7. **Add the New Theory**  
   Implement the new SMT theory in the SMT-LIB converter:  
   [Source Link](https://github.com/esbmc/esbmc/blob/master/src/solvers/smtlib/smtlib_conv.cpp#L359)

8. **Adjust Simplification Logic**  
   Modify the expression simplification logic as needed:  
   [Source Link](https://github.com/esbmc/esbmc/blob/master/src/util/simplify_expr.cpp#L2577)

9. **Check for Additional Operations**  
   Verify whether new operations need to be added in the SMT converter:  
   [Source Link](https://github.com/esbmc/esbmc/blob/master/src/solvers/smt/smt_conv.h#L288)
