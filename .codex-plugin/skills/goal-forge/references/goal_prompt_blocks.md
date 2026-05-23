# Goal Prompt Blocks

Use this structure for `GOAL.md`.

```xml
<goal>
[What the task produces. Specific, measurable, and execution-oriented.]
</goal>

<context>
[Files to read first, systems to inspect, and discovery commands if exact files are unknown.]
</context>

<constraints>
[Architecture rules, non-goals, risk boundaries, and anti-pattern fences.]
</constraints>

<scorecard>
[How the agent scores progress during the run: primary metric or checklist, passing threshold, regression checks, scoring command or inspection path, and stop condition.]
</scorecard>

<done_when>
[Concrete user-approved completion criteria. Tests, artifacts, behavior, and explicit non-regression checks.]
</done_when>

<feedback_loop>
[Fast representative checks to run while iterating: command/check, expected runtime, cadence, proxy validity, and slower escalation check.]
</feedback_loop>

<workflow>
[Step-by-step execution order. Identify which reads/searches can run in parallel and where verification gates belong.]
</workflow>

<working_memory>
[Markdown files the agent must create or maintain during long runs, such as PLAN.md, ATTEMPTS.md, and NOTES.md, plus when to update each one.]
</working_memory>

<human_control_surface>
[Optional. Human-visible status, task-specific knobs, sidecar inputs, and decision/resource gates the agent must reread before phase changes or strategic pivots.]
</human_control_surface>

<verification_loop>
[Commands and manual checks to run after major changes. Include fallback instructions when a check cannot run.]
</verification_loop>

<execution_rules>
[Stable Codex working rules for repo hygiene, edits, testing, and final communication.]
</execution_rules>

<output_contract>
[Expected final artifacts, final response shape, and completion signal.]
</output_contract>
```

Save only the XML block body in `GOAL.md`. The user can run or paste it after the Codex `/goal` command; do not include a standalone `/goal` line in the file unless the user explicitly asks for a paste-ready command transcript.

## Block Guidance

`<goal>` should describe the deliverable, not the process. Avoid vague verbs like "improve" unless paired with measurable behavior.

`<context>` should make the first turn efficient. Prefer concrete files. If the exact files are unknown, include discovery commands such as `rg "symbol"` or `rg --files`.

`<constraints>` should prevent shortcuts that could pass tests while violating intent. Include non-goals here.

`<scorecard>` should make loop evaluation explicit. Name the primary metric or checklist, the pass threshold, the regression checks, the exact scoring command or inspection path, and the stop condition. If the score depends on model judgment, provide a checklist or rubric so the judgment is auditable.

`<done_when>` is the termination contract. It must be concrete enough that the agent can decide whether to call the task complete.

`<feedback_loop>` should define the fastest useful check the agent can run repeatedly while working. Include expected runtime, when to run it, why it is representative enough, and what slower check must run before final completion.

`<workflow>` should sequence the work in phases: inspect, plan, implement, verify, refine, final review.

`<working_memory>` should be present for goals that may run for hours, involve repeated experiments, or require many context compactions. Prefer `PLAN.md` for the current plan, `ATTEMPTS.md` for tried approaches and results, and `NOTES.md` for chronological discoveries and blockers. For short linear goals, explicitly say working-memory files are not required.

`<human_control_surface>` should be present when the user may need to monitor, steer, pause, or constrain a long-running goal without rewriting the whole goal. Keep it minimal and task-specific. Include where the agent reports status, which knobs the user may edit, when the agent must reread them, sidecar inputs it may consume, and which strategic pivots require explicit approval. Omit this block for short linear tasks where it would add ceremony.

`<verification_loop>` should include focused checks first, then broad checks. If manual QA is required, specify what evidence is sufficient.

`<execution_rules>` carries stable agent behavior, such as preserving unrelated changes and using `apply_patch`.

`<output_contract>` should say what files or artifacts must exist and how concise the final response should be.
