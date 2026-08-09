# Contributing — how we work in this repo

Team guide for Tom + Chris. The goal: production-style habits, small revertible
units of work, and a `main` branch that always works. If any of this is new, just
follow the recipes below verbatim — they cover 95% of what you'll do.

## The golden rules

1. **`main` is never broken.** Nobody commits directly to `main`; everything
   lands through a pull request with green CI.
2. **One feature = one branch = one PR.** Small and focused: if a PR needs the
   word "and" twice in its title, it's probably two PRs.
3. **Merged = done and working.** A PR merges only when it runs/tests clean and
   the description says how to verify it.
4. **Anything can be backed out.** We squash-merge, so every feature is exactly
   one commit on `main` — reverting a feature is one command.

## Day-to-day recipe

```bash
# 0. Start from fresh main
git checkout main
git pull

# 1. Branch for your feature (prefix: feat/ fix/ docs/ chore/)
git checkout -b feat/replay-producer

# 2. Work in small commits — commit messages say WHAT changed and WHY
git add <files>            # add the files you actually meant to change
git commit -m "Add replay producer that re-emits recorded events with original timestamps"

# 3. Keep your work synced (only matters if main moved while you worked)
git fetch origin
git rebase origin/main     # replays your commits on top of latest main

# 4. Push and open the PR
git push -u origin feat/replay-producer
gh pr create --fill        # or use the GitHub web UI

# 5. When CI is green and it's reviewed → squash-merge
gh pr merge --squash --delete-branch
```

## What goes in a PR description

Three lines are enough:

- **What:** one sentence on what this adds/changes.
- **Why:** one sentence of context (link the TEAM_PLAN task, e.g. "task 2.5").
- **How to verify:** the command you ran and what output proves it works.

## Reviews

- Tag the other person on PRs that touch shared contracts (schemas, topics,
  interfaces between your components) — those need four eyes.
- Plumbing/docs/self-contained PRs: self-merge when CI is green is fine — note
  it in the PR ("self-merging, scaffolding only"). We're two people on a
  deadline; review where it pays, don't ritualize it.
- Review = pull the branch, run the verify command, skim the diff for surprises.

## CI and testing (QC rules)

Every PR runs GitHub Actions (`.github/workflows/ci.yml`): `ruff check` (lint),
`ruff format --check` (formatting), and `pytest` (full test suite). Red CI = fix
before merge, no exceptions — branch protection enforces it. Run locally before
pushing:

```bash
uv run ruff check . && uv run ruff format . && uv run pytest
```

**Every feature PR ships with tests.** New logic gets unit tests in `tests/`
(same PR, not "later"); pure functions and geometry/window/contract logic
especially — that's where our grade-critical correctness lives. Plumbing that
can't be unit-tested (live streams, cloud calls) gets a documented manual
verification step in the PR description instead.

## Reverting a feature

Every squash-merged PR is one commit on main:

```bash
git log --oneline          # find the commit for the bad feature
git checkout -b fix/revert-replay-producer
git revert <sha>
git push -u origin fix/revert-replay-producer && gh pr create --fill
```

## Repo hygiene (course + team rules)

- **Never commit:** `.env`, credentials, API keys, camera stream URLs or other
  external endpoints (config goes through `.env`; see `.env.example`), model
  weights, captured video/frames, anything in `internal-docs/`.
- **Always update `TEAM_PLAN.md`** when you claim or finish a task (that file is
  our contribution record for the rubric — it can ride along in a feature PR).
- Keep dependency changes visible: if you `uv add` something, say so in the PR
  description (`pyproject.toml` + `uv.lock` + `requirements.txt` change together
  via `uv export --no-hashes --no-dev -o requirements.txt`).

## Git glossary (30 seconds)

- **branch** — your private line of work; nothing you do on it affects `main`.
- **rebase on main** — pretend you started your branch from today's `main`;
  keeps history linear and merges conflict-free.
- **squash merge** — all your branch commits become ONE commit on `main`.
- **PR (pull request)** — a proposed merge + the place CI runs and review happens.
