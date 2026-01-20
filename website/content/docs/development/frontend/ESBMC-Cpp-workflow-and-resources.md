---
title: CPP Workflow and Resource
---

For feature support tracking, we have two documentations:
- Core language features: https://github.com/esbmc/esbmc/wiki/ESBMC-Cpp-Support for any completed core language feature.
- Library: https://github.com/esbmc/esbmc/wiki/OM-Workload-Estimate-and-Tracking for any completed library support (i.e. OMs).

For issue tracking, we used the strategies below:
- Label each C++ issue with “C++”
- Use the filter “is:issue is:open label:C++” to see a list of C++ pertaining issues
- For a major feature or bug fixes, we usually raise an umbrella bug to break it into smaller tasks and work our way up to the point when we get a good passing rate, e.g.
  - https://github.com/esbmc/esbmc/issues/1156  (WIP)
  - https://github.com/esbmc/esbmc/issues/989  (Done)
- To implement the best Agile practices, Github provides the ‘Projects’ tab to create and plan you iterations, e.g. https://github.com/esbmc/esbmc/projects/2

We usually follow a test-driven development approach and start the feature support with an issue to create a test suite, e.g. https://github.com/esbmc/esbmc/issues/1322.

The old test suites are not always the best to start with as the test cases might contain a mix of advanced features. Please feel free to add new test suites and design your own test cases when you start to work on a feature support. To save the time and effort, it is recommended to split an existing test cases that contains a mix of language features into multiple simple test cases that contain only one feature each.  