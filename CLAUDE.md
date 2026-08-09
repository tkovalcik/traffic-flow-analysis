# Project rules for AI-assisted work

Read CONTRIBUTING.md first — it defines the workflow. Non-negotiables:

- **Never commit directly to main.** Small per-feature branch → PR → green CI →
  squash-merge. One feature per PR so anything can be reverted in one command.
- **Every feature PR ships with tests** for the logic it adds. Lint + format +
  full test suite run in CI on every PR; red CI never merges.
- **No secrets or external endpoints in the repo** — camera stream URLs, API
  keys, and cloud identifiers go through `.env` (see `.env.example`). Sources
  are referenced in prose only (e.g. "Caltrans District 4 public camera map").
- **internal-docs/ is gitignored** — team-private notes live there, never in
  commits.
- **TEAM_PLAN.md is the work ledger** — claim tasks with initials, mark them
  done in the PR that completes them. It doubles as our graded contribution
  record.
- The core Kafka path (producer setup, consumer loop, window logic) is
  hand-written by the team for learning purposes — build around it, don't
  ghost-write it.
