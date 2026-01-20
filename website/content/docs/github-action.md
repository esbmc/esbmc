---
title: GitHub Action
weight: 11
---

The [esbmc-action](https://github.com/esbmc/esbmc-action) is a custom GitHub action which allows users to verify files on a GitHub repository using ESBMC. The user can either manually specify files to be verified, or, they can not specify any files in order to verify C and C++ files with a git diff between the two most recent commits. Note that the git diff functionality of this action relies on the user using appropriate naming conventions for their files: when using a git diff, ESBMC will only run on files that end with the '.c' or '.cpp' extension.

This action is supported on Linux, Windows, and macOS runners. For a more detailed guide on how to use this action, see the [esbmc-action repository's README](https://github.com/esbmc/esbmc-action/blob/main/README.md).

## Getting Started with Workflows:

For a very brief example of how to set up a workflow file which uses the esbmc-action and access its output on GitHub, see [this video](https://youtu.be/oWTRLMiT2XM). When using the git diff feature of this action, it is recommended to make the workflow trigger on a push, or something similar. The user can also use the esbmc-options input to pass command line options to esbmc on each run.

## Manually Specifying Files

To see how to manually specify files with this action (instead of using the git diff as above), see [this video](https://youtu.be/tLuGGm7HpsQ). This video also shows the use of the fail-fast and display-overhead options, which marks the workflow as failed when an ESBMC verification error occurs and displays the resource usage of ESBMC, respectively. If no files are specified for the 'files' input, this action will instead run off the git diff.

## Creating Artifacts

This action gives the user the option to create a GitHub artifact when using certain ESBMC options which generate files. To see how to do this, watch [this video](https://youtu.be/Zy2jR1MH1HU). The create-artifact option can be enabled to enable artifact creation, and the user can also specify the artifact-retention-days and artifact-compression-level. As shown in the video, the artifact can be accessed with an id and url for later use.

## Manual Checkout

This action will automatically checkout at it's start, but this may be disabled in the case that the user wishes to modify the contents of the runner before running ESBMC. If the user disables this option (sets it to 'n'), then they still must checkout before this action (with a fetch-depth of 2 if they wish to use a git diff). Watch [this video](https://youtu.be/LI4Zp1tNMJE) for a demonstration of a manual checkout.

