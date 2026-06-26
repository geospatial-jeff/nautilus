---
name: writing-docs
description: Use when writing or editing any documentation in this repo — READMEs, DESIGN.md, the implementation plan, reference docs, markdown files, module and function docstrings, and code comments. Defines what each kind of doc is for and how to write it.
---

# Writing docs

The documentation standards are the single source of truth in the project `CLAUDE.md` (§ "Writing
docs"), which is loaded every session — so they already apply to the docs you write while planning and
implementing, not just when this skill is invoked.

Invoke this skill to run a **focused pass**: review docs you have written or are editing against the
`CLAUDE.md` standards and revise where they diverge. Work through them in this order:

1. **Each doc has one job** — does each doc answer its one reader's one question, and nothing else?
2. **The funnel** — does each fact live at the highest altitude where it is that reader's concern, and
   never repeat below it?
3. **The rules** — explain don't list; concrete over abstract; justify every fact for its reader;
   simpler is not shorter; plain English with no decoration; match the code's words; never hand-edit
   generated docs (`docs/telemetry-reference.md`).

Read the full standards (including the worked example) in `CLAUDE.md` before reviewing.
