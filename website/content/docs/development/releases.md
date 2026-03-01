---
title: Releases
---

This is a guide on preparing and releasing new ESBMC versions.

> [!WARNING] Warning
>
> This guide is meant for developers of ESBMC with permissions for releasing new
> versions. It will not work if you don't have appropriate permissions.

To create a new ESBMC release on GitHub, follow these steps:

{{% steps %}}

### Create a Branch for the Release

Create a branch appropriately named for this release and switch to it, in this
case:

```sh
git switch -c v8.1
```

### Update the Release Notes

Update the release notes in `/scripts/release-notes.txt` to reflect the changes
included in the new version.

> [!TIP] Tip
>
> You can use Claude or any other LLM for this, just use this command to get the
> differences from the last tag
> `git log v8.0..master --oneline > /tmp/commits-for-release.txt` (in this case
> being `v8.0`) and instruct it to update the file for the next version.

### Update the Project Version

Update the project version by modifying `/CMakeLists.txt`. Specifically the
macros `ESBMC_VERSION_MAJOR`, `ESBMC_VERSION_MINOR`, `ESBMC_VERSION_PATCH` to
include the desired version number.

### Commit and Push to the Branch

Commit the changes and push them to the branch created earlier. Then submit a
PR. Name the commit appropriately, in this case `update: bump version to v8.1`.

```sh
git add scripts/release-notes.txt CMakeLists.txt
git commit -m "update: bump version to v8.1"
git push
```

### Create the PR

> [!NOTE]
>
> You can do this from the GitHub UI as well. But for convenience I show the
> commands.

Make sure to change the title and body to your version number.

```sh
gh pr create \
    --title "update: bump version to v8.1" \
    --body "Bump version to v8.1 and update release notes."
```

### Await PR to be Merged

Allow 2 other maintainers to review your PR as usual, approve and merge it.

### Tagging the Commit

Create a tag for the commit with the version change, make sure to name it
according to the version you're releasing. In this case `v8.1`:

```sh
git tag v8.1
```

Push the tag to origin:

```sh
git push --tags
```

### Creating the Release Artifacts

You need to get the build artifacts that will be uploaded. Run the Release job
on the `master` branch. Download the artifacts and prepare to upload them in the
next step.

### Creating the Release on GitHub

After pushing the tag, GitHub will allow you to create the official release
associated with that tag through the repository’s Releases page.

1. Select the tag you created and title the release according to your version
   "Release 8.1" (note the missing v, for consistency with past releases).
2. Write the release description, which should just be a markdown version of the
   content written in `/scripts/release-notes.txt` from earlier.
3. You will need to upload the release artifacts created in the last step.

> [!TIP]
>
> You can use Claude or any other LLM to prepare the release description.

{{% /steps %}}
