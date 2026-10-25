# Memory Bank

This is the shared knowledge base for the HuBenchMarks project. Both agents (Claude and DeepSeek) read from these files; each agent writes only to their own directory.

## Read these two first

| File | Use it for |
|---|---|
| **`claude/master-results.md`** | **Authoritative numbers.** Every accuracy, subcategory and significance test. If anything else disagrees, this wins |
| `claude/activeContext.md` | Orientation: current status, the one open job, hard-won facts, and the failure modes to watch for |

`claude/progress.md` is a condensed scoreboard. Dated planning documents and
point-in-time health checks live in `archive/memory-bank-snapshots/` — they are
superseded and should not be read for current status.

## Structure

```
memory-bank/
├── shared/              # Shared reference — both agents read, rarely write
│   ├── projectbrief.md    Core mission, constraints, scope
│   ├── productContext.md  Why this project exists, how it works
│   ├── systemPatterns.md  Architecture, design patterns
│   ├── techContext.md     Technologies, cluster paths, conda envs
│   └── paper-audit.md     Model cross-check against published papers
├── claude/              # Claude's working state (only Claude writes here)
│   ├── activeContext.md   Current focus, job queue, open issues
│   └── progress.md        What Claude has completed / is working on
└── deepseek/            # Other agent's working state (only they write here)
    ├── activeContext.md   Current focus, job queue, open issues
    └── progress.md        What the other agent has completed / is working on
```

## Working Rules

1. **Write only to your own directory** (claude/ or deepseek/)
2. **Read shared/ and the other agent's directory** for context
3. **Edit shared/ sparingly** — only for new models, new paths, new patterns
4. **Keep activeContext.md current** — other agent depends on it to avoid duplicate work
5. **No file locking** — coordination is via reading each other's status, not locking
