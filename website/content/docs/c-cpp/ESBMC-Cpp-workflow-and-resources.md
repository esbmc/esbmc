---
title: C++ Workflow and Resources
---

This page is for ESBMC maintainers working on the C++ frontend. For the
user-facing feature reference, see [C++ Support](./supported-features) and
[C++ Limitations](./limitations).

For feature support tracking, we have two documentations:
- Core language features: [C++ Support](./supported-features) for any completed core language feature.
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

## Benchmark tracking

The C++ regression suites live under `regression/esbmc-cpp*`, split by standard
(`esbmc-cpp` for C++98/03, then `esbmc-cpp11`, `esbmc-cpp14`, `esbmc-cpp17` and
`esbmc-cpp20`). Current pass rates come from
[CI](https://github.com/esbmc/esbmc/actions) rather than being tracked by hand;
the `KNOWNBUG`-marked cases are catalogued under the umbrella issue
[#4403](https://github.com/esbmc/esbmc/issues/4403).

The stats below are generated from the benchmark logs of the GitHub workflow
"Run a Benchmark".

Error signatures:

```bash
egrep "Assertion|ERROR" * -rn | egrep -v "//" | cut -d':' -f3- | sort | uniq -c
```

Passes and failures:

```bash
egrep "VERIFICATION FAILED|VERIFICATION SUCCESSFUL" * -rn | rev | cut -d ':' -f 1 | rev | sort | uniq -c
```

### Bulk test-case edits

Kept for reference. Fix the include path for the Linux CIs:

```bash
egrep "\-I ~/libraries" . -rl | xargs sed -i 's/-I \~\/libraries/-I \/__w\/esbmc\/esbmc\/src\/cpp\/library/g'
```

Tag every test case in a suite:

```python
from pathlib import Path
for path in Path('./').rglob('test.desc'):
    print(path)
    f = open(path,'r')
    lines = f.readlines()[:-1]
    lines.append("<item_10_mode>KNOWNBUG</item_10_mode>" + "\n")
    lines.append("</test-case>")
    f.close()
    f = open(path,'w')
    f.writelines(lines)
```
