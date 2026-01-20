---
title: Setup
# prev: /docs
# next: /docs/usage
weight: 1
---

<p>To install ESBMC on your machine, you should download the latest binary for Linux and Windows OSs from <a href="https://github.com/esbmc/esbmc/releases">GitHub</a> and save and unzip it on your disk.</p>

<p>Once the user unzips the release, they should read the license before running ESBMC. The ESBMC distribution is split into two directories:</p>

<ul>
    <li><code>bin</code>: contains a static-binary file of ESBMC;</li>
    <li><code>license</code>: contains the ESBMC, Z3 and Boolector licenses.</li>
</ul>

<p>If the user wants to use other SMT solvers (e.g., MathSAT, Yices, CVC4), we recommend checking out the ESBMC source code, which is hosted on <a href="https://github.com/esbmc/esbmc" target="_blank">GitHub</a>, and then follow the instructions in the <a href="https://github.com/esbmc/esbmc/blob/master/BUILDING.md" target="_blank">BUILDING </a>file.</p>