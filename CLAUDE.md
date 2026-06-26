# Nautilus — project guidance

Nautilus is a decentralized, entirely-streaming parallel compute framework. In a distributed system the
documentation carries as much of the design as the code does: invariants, why-not alternatives, and the
contracts between processes are mostly invisible in the code itself. A wrong or stale doc misleads every
future reader. So documentation is a first-class deliverable here, not a cleanup pass.

## Documentation is part of every change

Apply the standards below from the **first draft** — when you write a plan (`DESIGN.md`,
`IMPLEMENTATION_PLAN.md`, or a working plan file) and when you write the initial module/function
docstring, inline comment, or reference-doc entry for new code. Do not draft prose loosely and fix it
later; write it to standard the first time. When a plan stages a deliverable, the docstrings, comments,
and reference/glossary updates that deliverable needs are part of it — name them in the plan and write
them as you build, not after.

## Writing docs

Every doc answers one question for one reader, and says only what the code can't. The code already
shows *what* it does and *how*; documentation exists for the *why*, the *intent*, the *how-to-use*, and
the *what-not-to-do*. A sentence that restates the code or the file tree is deletable.

### Each doc has one job

| Doc | Reader | The question it answers |
|---|---|---|
| README | just found the project | What is this, why should I care, how do I run it? |
| DESIGN.md | about to build on or change it | Why is it built this way? |
| IMPLEMENTATION_PLAN.md | building it now | What's done, what's next? |
| Reference docs (glossary, CLI, telemetry) | looking one thing up | What does this term / command / metric mean? |
| Module docstring (top of file) | about to edit this file | What is this file's job, and what invariants must I not break? |
| Function / class docstring | calling it | What's the contract — arguments, result, gotchas? |
| Inline comment | reading this exact line | Why this, instead of the obvious thing? |

### The funnel

A fact enters at the highest altitude where it is that reader's concern, and never repeats below.

- The **idea** lives at the top (README, DESIGN.md).
- The **mechanism** lives in the docstring of the code that implements it.
- The **line-level reason** lives in an inline comment.

Telemetry shows all three: that nautilus emits facts and not verdicts is a design decision, so it
lives in DESIGN.md. That the recorder is single-writer and lock-free is a property of that code, so it
lives in `recorder.py`'s docstring. That one line skips a catalog lookup is local, so it lives in an
inline comment there. None of the three repeats anywhere else.

### Rules

- **Explain, don't list.** Naming the modules a layer contains is a directory listing the file tree
  already gives you. Say what the layer *does*.
- **Concrete over abstract.** "counter, gauge, histogram" beats "instruments." Don't swap a familiar
  word for a jargon umbrella to save space — that raises what the reader must already know.
- **Justify every fact for its reader.** Don't keep a detail just because it was in the sentence you
  are editing. If this reader doesn't need it, cut it or move it down the funnel.
- **Simpler is not shorter.** The goal is the least a reader must already know to understand it, not
  the fewest words. A short sentence built from undefined jargon is worse than a longer plain one.
- **Plain English, no decoration.** No metaphors or slogans ("cheat-sheet", "first-class", "under the
  hood", "ships the data agents use to build it"). State the fact.
- **Match the code's words.** Use the exact names the code uses; do not substitute synonyms.
- **Don't edit generated docs.** `docs/telemetry-reference.md` is generated from the catalog — change
  the source and regenerate, never hand-edit the output.

### Example

This is a module list with adjective glosses; it only parses if you already know the code, and it
puts file-tree facts into a doc whose reader wants the *decision*:

> the instruments (`model`), the metric `catalog` (the frozen source of truth …), the per-actor
> `recorder` (single-writer, lock-free, zero-cost when off), and the `registry`

At the design level, explain the decision instead and let the implementation details live in the code:

> Instrumentation on the hot path only records raw numbers — it never assembles the report — so a run
> pays as little as possible for telemetry. import-linter forbids the per-record code from importing
> the report layer, so report-building can't creep onto the hot path.
