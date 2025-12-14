---
title: CI-CD
---

ESBMC repository relies on [GitHub Actions](https://github.com/features/actions) for all its CI/CD needs. The workflows files are all inside the `.github/workflows` and are described as follows:

- **benchexec**: This workflow does a run of [benchexec](https://github.com/sosy-lab/benchexec) in a given branch. It is only used by manual dispatch.  
- **build**: This is the main build workflow, which is invoked for every push/PR in ESBMC. It works by running fuzzing and unit tests and by: (1) configuring a build environment for macOS, Linux, and Windows; (2) Building the system and saving the results as artifacts; (3) Running the regression and unit tests.  
- **linter**: This applies linters over the code.
- **release**: Similar to the build, but generates release builds of ESBMC and creates a new release at the GitHub page.
- **sanitizers**: Does a quick build of ESBMC with clang sanitizers enabled.