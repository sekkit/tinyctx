# tinyctx custom chat templates

## qwen3.6-barubary-hermes-json-v1.jinja

A Hermes-JSON variant of the user's custom Barubary v1 template (which the user supplied in chat). All 21 fixes from the original are preserved verbatim. The only deliberate change is the **tool-call format** the model is instructed to emit:

| | Original Barubary v1 | This variant |
|---|---|---|
| Format | qwen-pythonic XML | Hermes JSON |
| `<tool_call>` body | `<function=NAME><parameter=KEY>VAL</parameter></function>` | `{"name":"NAME","arguments":{"KEY":"VAL"}}` |
| LMStudio's parser handles it natively? | No (we needed `tinyctx.tool_call_translator`) | Yes (llama.cpp's built-in Hermes parser) |
| Off-distribution risk for qwen3-coder family? | None (this IS its training format) | Real (model was tuned on the pythonic format) |

## How to apply

1. The file is also at `~/Desktop/qwen3.6-barubary-hermes-json-v1.jinja` for quick access.
2. Open LMStudio → load your qwen3.6-* model.
3. Open the model's settings panel (gear icon next to the model in the loaded-models list).
4. Navigate to **Prompt Template** (or "Prompt Format" / "Chat Template" depending on LM Studio version).
5. Switch to the **custom / jinja** template editor and paste the entire file contents.
6. Save & reload the model.

## When NOT to switch

- You're using `tinyctx` (this repo) — its `tool_call_translator` already handles the original pythonic XML deterministically without changing the model's distribution. **Verified end-to-end with codex CLI 0.125 + LMStudio + qwen3.6-27b-crack on 2026-05-06.**
- You're chasing top quality on tool-heavy agent loops — qwen3-coder's RL training is on pythonic XML; switching costs some accuracy.

## When to switch (where Hermes-JSON helps)

- You're not running the tinyctx proxy and want LMStudio to emit structured `function_call` items directly to clients like raw codex.
- You're running an agent client that doesn't have its own translator and only understands OpenAI structured tool calls.
- You're integrating with a third-party tool that expects standard Hermes format.

## A/B test recipe

```bash
# 1. Save the current template (whatever's in LMStudio settings now) to a file
#    so you can revert.
# 2. Load this Hermes-JSON variant.
# 3. Run an identical agent task on both and compare:
#    - tool call success rate
#    - argument fidelity (especially: multi-line strings, paths with quotes,
#      deeply-nested JSON args, code blobs inside arguments)
#    - end-to-end task completion rate

# Useful probe (with model loaded and LMStudio server up at :1234):
curl -s -X POST http://127.0.0.1:1234/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-crack","stream":false,
       "input":[{"role":"user","content":[{"type":"input_text",
                 "text":"List the files in /tmp using the ls tool."}]}],
       "tools":[{"type":"function","name":"ls",
                 "description":"List files in a directory.",
                 "parameters":{"type":"object",
                               "properties":{"path":{"type":"string"}},
                               "required":["path"]}}],
       "tool_choice":"auto","text":{"format":{"type":"text"}}}' \
  | python3 -m json.tool

# Compare output[]:
#   * Original Barubary template → output[0].type=="message", text contains XML
#   * Hermes-JSON variant        → output[0].type=="function_call" (if llama.cpp parses it)
#                                  or output[0].type=="message" with JSON-in-XML text
#                                  (if llama.cpp's parser doesn't accept the format —
#                                  in which case keep the original template + use tinyctx)
```

## File map

```
templates/
├── README.md                                    # this file
└── qwen3.6-barubary-hermes-json-v1.jinja        # the Hermes-JSON variant (328 lines)

~/Desktop/
└── qwen3.6-barubary-hermes-json-v1.jinja        # mirror for easy LMStudio paste
```
