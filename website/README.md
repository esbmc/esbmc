# ESBMC Website

This directory represents the files for ESBMC's website. They get built by the
GitHub action /.github/workflows/pages.yml and published to GitHub pages when
the master branch is update.

To run it locally, install [Hugo](https://gohugo.io/) and run:

```sh
hugo server -D
```

Remember to specify `-D` as it will show articles flagged as `draft`.

As you can see, the "text" content is in the `content` folder, and is defined as
markdown files with some meta properties defined at the top. Here is a non-comprehensive
list of important ones:

* draft: set to true to not show in the built site
* author: displays the author of a specific article (in `/news` only)
* tags: list of tags that will be ascociated with a specific article (in `/news`
only)
* date: automatically generated and is **important** as it defines at which place
the article will be shown.

## Creating New Articles

To create new articles simply specify the directory you want to create them in:

>Docs article
```sh
hugo new content/docs/example.md
```

>News article
```sh
hugo new content/news/esbmc-version-8.md
```

A date value will be assigned and the article will be marked as draft. **Remember
to set the `draft` flag to `false` when ready to publish.**

## Static Assets

There is a `static` folder where assets can be comitted into, it's generally
discouraged to commit large assets to that directory. Use 
[git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage) 
to link large files to the `static` dir instead.

## Hextra Template

The Hextra template we are using is quite versatile, please the see 
[docs](https://imfing.github.io/hextra/docs/getting-started/) for a complete
list of features.
