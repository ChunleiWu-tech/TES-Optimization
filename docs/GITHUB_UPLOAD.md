# GitHub upload guide

This repository is prepared for upload with the Git command line. Do not use browser drag-and-drop upload: the retained, validated CSV result tables exceed typical web-uploader limits.

1. Create an empty GitHub repository without adding a README, `.gitignore`, or licence through the GitHub interface.
2. Open PowerShell at this directory and run:

```powershell
git init
git add .
git status
git commit -m "Release reproducible V18 molten-salt TES co-design workflow"
git branch -M main
git remote add origin https://github.com/[ACCOUNT]/[REPOSITORY].git
git push -u origin main
```

3. Replace `[ACCOUNT]` and `[REPOSITORY]` with the owner and repository name. GitHub authentication is handled by your normal Git credential manager or the GitHub CLI.
4. Before selecting **Public**, complete `CITATION.md` and add the intended licence. Confirm that public release of all source-derived CSV data is permitted by the underlying sources.

The repository does not contain the submission Word documents, author identities, journal-specific material, or a copy of any cited publisher PDF.
