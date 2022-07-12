# Git & Github

- Basic concepts:
  - Repository (repo): a working folder
  - Commit:
    - a change of the content of a repo
    - all versions of a repo form a tree, where each node represents a commit
  - Branch:
    - a pointer to a certain commit
    - usually represented by the first 5 characters of the hash value
  
- How to copy a repository for your own use:
  - use `fork` instead of `clone`, creating a repository owned by you
  - `clone` your own repository to local device and edit it
  - note:
    - `fork` is the feature of Github, not Git itself
    - you need to do it on Github's webpage
  
- How to synchronize with other's modification on remote (e.g. the repository you fork)
  - `fetch` the remote repository to local device
  - examine whether it contains conflict with your repository
  - `merge` it with your repository
  - if you are confident that there is no conflict, you can do `pull` directly, which is `fetch + merge`
  
- How to contribute your update to the remote repo:
  - `fork` the remote repo and `clone`
  - edit it and `push` the change to your repo
  - create a pull request
  - wait for the owner to accept your pull request and `merge` it to the remote repo
  
- How to overturn a change:

  - Overturn and delete all commits that are not yet pushed to the origin:

    ```bash
    git reset --hard origin/branch
    ```

    where "origin" is the name of the remote (usually just "origin") and "branch" is the name of the branch.

    This command will overturn all changes not in the history of `origin/branch` and delete relavant commits.

  - Revert commits after pushing to the origin:

    - Note: once a commit is pushed to the origin, it should not be deleted, since someone else may use that version as their basis; however it can be reverted while keeping the history intact

    - Reverting 3 most recent commits:

      ```bash
      git revert --no-commit HEAD~3..
      git commit -m "your message regarding reverting the multiple commits"
      ```

      Note:

      - Does NOT work for merging commits! (See [here](https://stackoverflow.com/a/1470452))
      - In git, `HEAD` is a pointer to the head of your current branch
      - Don't forget the two dots after the number! You can also change `HEAD~3..` to `HEAD~3..HEAD`

- `.gitignore`:

  - Usually we won't use `git` to track large binary files, which will be very space-consuming
  - https://github.com/github/gitignore provides many templates for the `.gitignore` file
  - If you want to ignore a file that is already tracked, you need to:
    - Untrack it:
      - `git rm --cached FILENAME` (or `git rm --cached -r FOLDER`)
      - will "delete" it from git repo but not from the disk

    - Modify your `.gitignore` file

- Working within VS Code: see https://docs.microsoft.com/en-us/learn/modules/use-git-from-vs-code/3-exercise-clone-branch

