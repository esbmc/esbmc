---
title: API Reference
---

ESBMC's C++ classes and methods are documented inline with
[Doxygen](https://www.doxygen.nl/) comments in the header files. The generated
API reference is published at
[esbmc.github.io/docs/api](https://esbmc.github.io/docs/api) and is rebuilt
automatically on every push to `master`.

> [!NOTE]
> Markup coverage is partial — only a limited number of classes carry Doxygen
> comments so far. Contributions that document more of the API are welcome.

## Build it locally

Install `doxygen` and `graphviz`, then generate the HTML from the repository
root. Both methods share the single `.doxygen` Doxyfile, so they produce the
same output as the published reference.

{{< tabs >}}

{{< tab name="Doxygen directly" >}}
```sh
doxygen .doxygen
```
The output is written to `docs/html`; open `docs/html/index.html`.
{{< /tab >}}

{{< tab name="Via CMake" >}}
```sh
cmake -Bbuild -S. -DBUILD_DOC=On
ninja -C build docs
```
This runs the same `.doxygen` configuration and writes to `docs/html`.
{{< /tab >}}

{{< /tabs >}}
