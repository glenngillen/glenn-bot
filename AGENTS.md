<!-- Synapse Agent Instructions v1.4.0 -->
<!-- DO NOT EDIT THE SYNAPSE-MANAGED SECTION BELOW -->
<!-- Last synced: 2026-07-23T06:16:11.852027+00:00 -->

**Note**: This project uses the native Synapse issue tracker for issue tracking.
Use `synapse issue` commands instead of markdown TODOs.
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

## Codebase Exploration Protocol

**IMPORTANT:** When exploring or understanding this codebase, follow this protocol:

### 1. Start with specs/README.md
Always read `specs/README.md` first. It provides:
- Architecture overview
- Quick Reference table of all specifications
- File locations and tech stack

### 2. Read Relevant Specs Only
Use the Quick Reference table to identify specs relevant to your task:
- Don't read all specs - only those related to your work
- Each spec contains file paths for implementation details

### 3. Targeted Code Reading
Only read source code when necessary:
- Read files explicitly mentioned in specs
- Read files directly required for your task
- Prefer reading specific functions over entire files

### 4. Do NOT Scan Entire Codebase
- Never try to ingest or review the whole codebase
- Never read files "just in case" they might be relevant
- Trust specs to guide you to relevant areas

### If specs/ Doesn't Exist
If this project has no `specs/` directory:
1. Fall back to traditional exploration
2. Create a task to establish specs for the project

## Issue Tracking with Synapse

**IMPORTANT**: This project uses the **Synapse native issue tracker** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why Synapse Issues?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Stored alongside code in the repository
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**
```bash
synapse issue list --status open --json
```

**Create new issues:**
```bash
synapse issue create "Issue title" --type bug|feature|task --priority 0-4 --json
synapse issue create "Issue title" --priority 1 --deps discovered-from:<parent-id> --json
synapse issue create "Subtask" --parent <epic-id> --json
```

**Claim and update:**
```bash
synapse issue update <id> --status in_progress --json
synapse issue update <id> --priority 1 --json
```

**Complete work:**
```bash
synapse issue close <id> --reason "Completed" --json
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

1. **Check ready work**: `synapse issue list --status open` shows available issues
2. **Claim your task**: `synapse issue update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `synapse issue create "Found bug" --priority 1 --deps discovered-from:<parent-id>`
5. **Complete**: `synapse issue close <id> --reason "Done"`
6. **Commit together**: Always commit issue state changes together with the code changes so issue state stays in sync with code state

### Important Rules

- Use Synapse issue tracking for ALL task tracking
- Always use `--json` flag for programmatic use
- Link discovered work with `discovered-from` dependencies
- Check open issues before asking "what should I work on?"
- Run `synapse issue <cmd> --help` to discover available flags
- Do NOT create markdown TODO lists
- Do NOT use external issue trackers
- Do NOT duplicate tracking systems

## Testing Standards

### Unit Test Purity Rules

Unit tests MUST be pure — no side effects that escape the test process:
- **No network calls** — No `fetch`, `http`, `net`, or real network I/O
- **No external programs** — No spawning `git`, `gh`, `node`, or child processes
- **No real filesystem** — Mock `fs` or use temp directories; never touch project files
- **Mock all external interfaces** — Use `vi.mock()` (or equivalent) for I/O modules

### Test Tiers

| Tier | Runs Where | Speed | Isolation |
|------|-----------|-------|-----------|
| Unit | Locally (TDD cycle + pre-commit) | <60s total | Fully mocked |
| Integration | CI only | <5min | Multi-module |
| E2E | CI only | <10min | Full workflow |

### TDD Mandate

1. Write failing tests first
2. Implement minimum code to make tests pass
3. Verify tests pass
4. Refactor while keeping tests green

### Property-Based Testing (PBT)

After the normal TDD cycle, for any newly added, signature/body-modified, or called
function that is a strong PBT candidate, you must add property tests.

**When your work adds or modifies a strong candidate, you must add property tests:**
- Parsers and transformers (structured text, config, markdown) → counters/flags consistent
- Sanitisation functions → output always satisfies the target format's rules
- Arithmetic aggregates (counts, percentages, indices) → numeric bounds hold
- Serialise/deserialise pairs → roundtrip: `parse(format(x)) == x`
- Functions receiving arbitrary external input → no-panic guarantee

**Weak candidates (skip PBT, unit tests are enough):**
- Dispatch logic (enum match arms, routing)
- I/O functions and network calls
- UI rendering logic

**Three-tier decision rule:**
1. Touched strong candidate → must add property tests
2. Weak candidate → no PBT, write unit tests instead
3. Unsure (external-input function) → add a simple no-panic property as a minimum

**Opt-out rule:** If you cannot state an invariant without naming a specific input,
skip PBT — but note the omission in a comment rather than silently omitting it.

**PBT scope:** apply to newly added, signature/body-modified, or strong candidates called
from new code. Do NOT backfill unrelated untouched functions elsewhere in the codebase.

**Rust convention (canonical example):**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Example-based tests here

    mod props {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn task_counts_are_consistent(content in arb_plan_content()) {
                let plan = parse_plan_content(&content, None);
                prop_assert_eq!(
                    plan.total_tasks,
                    plan.completed_tasks + plan.incomplete_tasks
                );
            }
        }
    }
}
```

**Per-language framework and placement:**
- Rust → `proptest` in a `mod props` block inside `#[cfg(test)]`
- Go → `rapid` in `TestProperties*` functions in `_test.go`
- TypeScript/Node → `fast-check` in a `describe("properties", ...)` block
- Python → `hypothesis` in a `class TestProperties` in the test module
- Swift → `PropertyBased` with `@Test`/`forAll` (Swift Testing); `XCPropertyBased` for XCTest

**Python failure persistence:** Do not commit `.hypothesis/`. When Hypothesis finds a
failure, add the shrunk case as a named example-based regression unit test, fix the
implementation, then verify both tests pass.

**Dependency installation:** Add the PBT framework as a dev/test dependency when you
write the first property test — not before. See `standards/common/TESTING_PROPERTIES.md`
for exact commands, check-before-adding instructions, and the dependency-free fallback.

### Running Tests

```bash
# During TDD cycle — scoped to current work:
synapse utils test-runner --scope unit --filter <pattern>

# Before commit — full unit suite:
synapse utils test-runner --scope unit

# Integration/e2e — CI only (never run locally in agentic loop)
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
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

