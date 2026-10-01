# Game ↔ ML protocol (v1)

- Transport: TCP, `127.0.0.1:5555` by default
- Framing: one JSON object per line (UTF-8, `\n`-terminated)
- Every request gets exactly one response, in order

## `ping`

```json
{"type": "ping"}
```
```json
{"type": "pong", "protocol": 1}
```

## `classify`

`strokes` is a list of strokes; each stroke is `[xs, ys]` in canvas pixels,
matching the Quick, Draw! simplified format.

```json
{"type": "classify", "strokes": [[[10, 40, 80], [12, 60, 15]]]}
```
```json
{"type": "classify_result", "top_k": [{"label": "bicycle", "prob": 0.82}, {"label": "car", "prob": 0.07}]}
```

## `chat`

`session` groups the turns of one negotiation. `item` is sent on the first turn.

```json
{"type": "chat", "session": "r42", "item": "bicycle", "text": "I want to sell this bicycle"}
```
```json
{"type": "chat_reply", "text": "How many years have you had it?", "offer": null}
```

On the final turn `offer` is filled in, or `refused` is set:

```json
{"type": "chat_reply", "text": "...", "offer": {"price": 2800, "currency": "TWD"}}
```
```json
{"type": "chat_reply", "text": "Sorry, I can't buy a protected animal.", "offer": null, "refused": true}
```

## `error`

```json
{"type": "error", "message": "unknown message type: 'foo'"}
```
