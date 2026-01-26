<!-- Synapse Agent Instructions v1.0.0 -->
<!-- DO NOT EDIT THE SYNAPSE-MANAGED SECTION BELOW -->
<!-- Last synced: 2026-01-12T04:39:33.079Z -->

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads)
for issue tracking. Use `bd` commands instead of markdown TODOs.
See the Issue Tracking section below for workflow details.

# Agent Instructions

This file provides instructions for AI agents working on this project.

## Worktree Enforcement (CRITICAL)

**ALL development MUST happen in worktrees, NEVER in the main repository directory.**

### How to Verify You're in a Worktree

Before making ANY code changes, verify your working directory:

```bash
# Check if you're in a worktree (should see "gitdir:" content)
cat .git
# If .git is a FILE containing "gitdir:", you're in a worktree
# If .git is a DIRECTORY, you're in the main repo - STOP IMMEDIATELY
```

Alternatively, check the path:
- Correct: `.worktrees/<issue-id>/` (e.g., `.worktrees/synapse-abc123/`)
- Wrong: The main project directory (e.g., `/Users/*/Development/*/glenn-bot/`)

### If You're in the Main Repository

**STOP IMMEDIATELY. Do not make changes.**

1. Exit the current session
2. Report the error to the user
3. Wait for proper worktree setup before continuing

### Why This Matters

- Worktrees enable concurrent development on multiple issues
- The main repository must stay on the `main` branch and clean
- Each issue gets its own worktree with its own feature branch
- This prevents merge conflicts and keeps work isolated

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**
```bash
bd ready --json
```

**Create new issues:**
```bash
bd create "Issue title" -t bug|feature|task -p 0-4 --json
bd create "Issue title" -p 1 --deps discovered-from:bd-123 --json
bd create "Subtask" --parent <epic-id> --json  # Hierarchical subtask (gets ID like epic-id.1)
```

**Claim and update:**
```bash
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

**Complete work:**
```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`
6. **Commit together**: Always commit the `.beads/issues.jsonl` file together with the code changes so issue state stays in sync with code state

### Important Rules

- Use bd for ALL task tracking
- Always use `--json` flag for programmatic use
- Link discovered work with `discovered-from` dependencies
- Check `bd ready` before asking "what should I work on?"
- Run `bd <cmd> --help` to discover available flags
- Do NOT create markdown TODO lists
- Do NOT use external issue trackers
- Do NOT duplicate tracking systems

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

<!-- END SYNAPSE-MANAGED SECTION -->
<!-- Project-specific instructions below this line will be preserved during sync -->


