# Getting this into Git and onto GitHub

One-time setup, about 10 minutes. Do this from the folder where you unzipped
this project (suggestion: move it somewhere permanent first, like
`~/Projects/reformulation-assurance` — not Downloads).

## 0. Decide your identity first

Commits permanently embed an author name and email. Decide now whether you are
launching under your real name or a handle, then configure git to match:

```bash
git config --global user.name "Your Name or Handle"
git config --global user.email "you@example.com"
```

Privacy tip: GitHub can give you a no-reply address so your real email never
appears in commits. GitHub → Settings → Emails → check "Keep my email
addresses private", then use the `...@users.noreply.github.com` address shown
there in the command above.

Also fill in the copyright line in `LICENSE` (currently a placeholder).

## 1. Create the repository locally

```bash
cd path/to/reformulation-assurance
git init
git add .
git status
```

Before committing, check `git status` output: you should NOT see `data/`,
`.env`, `.venv/`, any `*.db` file, or `.artifact_key` in the list. The
`.gitignore` handles this — if any of those appear, stop and ask.

```bash
git commit -m "Reformulation Assurance v0.6.3 — first public commit"
```

## 2. Create the GitHub repository

On github.com: New repository → name it (e.g. `reformulation-assurance`) →
Public → do NOT add a README/license/gitignore (you already have them) →
Create. Then connect and push:

```bash
git remote add origin https://github.com/YOURUSERNAME/reformulation-assurance.git
git branch -M main
git push -u origin main
```

## 3. From now on

Never make patch zips again. The loop is:

```bash
git add -A
git commit -m "what changed, in one line"
git push
```

Your Downloads folders (`reformulation_assurance_v06`, the patch and hotfix
folders) are now historical — the repo is the single source of truth. Keep
`data/` where it is; it stays local and untracked.
