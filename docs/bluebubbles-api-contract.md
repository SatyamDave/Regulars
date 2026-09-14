# BlueBubbles API contract — verification pass

Verifies every BlueBubbles REST/webhook assumption documented in `regulars/bot/transport.py` and
`regulars/features/ingest.py` against BlueBubbles' actual, current API. Prompted by the original
bot-runtime agent's own flag: those field shapes were "from my knowledge of its REST API, not
verified against a live payload."

## Sources consulted (primary, in order of authority)

1. **`BlueBubblesApp/bluebubbles-server` GitHub source**, commit `f2e2286` (cloned 2026-09-14,
   `package.json` version `1.9.9`) — the server's actual route table, validators, response types,
   and ORM entities. This is ground truth: it is what runs on the user's Mac.
   - Route table: `packages/server/src/server/api/http/api/v1/httpRoutes.ts`
   - Response shape: `packages/server/src/server/types.ts` (`MessageResponse`, `HandleResponse`,
     `ChatResponse`, `AttachmentResponse`)
   - Serializer: `packages/server/src/server/api/serializers/MessageSerializer.ts`
   - Webhook dispatch: `packages/server/src/server/index.ts` (`handleNewMessage`,
     `WebhookService.dispatch`)
   - Tapback code table: `packages/server/src/server/databases/transformers/MessageTypeTransformer.ts`
   - Raw-row decoder (used by the live poller that feeds webhooks):
     `packages/server/src/server/databases/imessage/entity/decoders/MessageDecoder.ts`
2. **BlueBubbles' official Postman collection**, "BlueBubbles" (`documenter.getpostman.com/view/765844/UV5RnfwM`,
   collection id `765844-d2fd54d9-6696-4399-9dcb-0246f14d4227`, fetched 2026-09-14) — this is
   BlueBubbles' own published API reference, including one saved live example response
   (`Chat / Get Chat Messages`) that is a real captured payload, not a schema description.
3. **`mautrix/imessage`** (Matrix.org's production iMessage bridge, commit `300ba6d`), used only to
   cross-check community-level facts not stated in BlueBubbles' own docs/source comments (chat GUID
   format, tapback-target GUID prefix stripping) — independent second implementation talking to the
   same BlueBubbles API.
4. `docs.bluebubbles.app` (official docs site) — thin on specifics; used only for the event-name
   list and to confirm the `{status, message, data}` response envelope and the `guid`/`password`/
   `token` auth aliasing.

No fabricated payloads were used as evidence. Where I could not find a live example, I say so.

## Per-assumption verdicts

### Authentication

- **Assumption in code**: `?password=<server password>` query parameter.
- **CONFIRMED.** `packages/server/src/server/api/http/api/v1/middleware/authMiddleware.ts`:
  `const token = params?.guid ?? params?.password ?? params?.token`. All three names work; the
  Postman collection's own example URLs use `?password={{password}}` for every endpoint, so
  `password` is not just an accepted alias, it's the one BlueBubbles documents. No change needed.

### Endpoints

| Assumption in `transport.py` | Verdict | Evidence |
|---|---|---|
| `POST /api/v1/message/text` `{chatGuid, tempGuid, message, method}` | **CONFIRMED** | `httpRoutes.ts` line ~415 (`prefix: "message"`, `path: "text"` → `MessageRouter.sendText`); `sendText` destructures exactly `tempGuid, message, method, chatGuid` from body (`messageRouter.ts:237-241`); `MessageValidator.sendTextRules` requires `chatGuid`, `message`, allows `method: apple-script\|private-api` (`messageValidator.ts:69-79`). Postman: `POST {{host}}/api/v1/message/text`. |
| `POST /api/v1/chat/new` `{addresses, message}` | **CONFIRMED** | `httpRoutes.ts` (`prefix: "chat"`, `path: "new"` → `ChatRouter.create`); `ChatRouter.create` reads `addresses, message, method, service, tempGuid, ...` (`chatRouter.ts:187-207`); `ChatValidator.createRules` requires `addresses` (array) (`chatValidator.ts:43-51`). Postman: `POST {{host}}/api/v1/chat/new`. |
| Chat creation requires an initial message | **CONFIRMED (behavioral, not enforced by validator)** | Not enforced as `required` in `ChatValidator.createRules` (only typed `string`), but `ChatInterface.create` is documented server-side and in community tooling as needing a first message to seed the iMessage chat row; sending one (as the code does via `voice.CHAT_OPENED`) is safe either way. Response `data.guid` is confirmed as the new chat id — `ChatRouter.create` returns `ChatSerializer.serialize({chat})`, and `ChatResponse.guid` is a top-level field (`types.ts:92`). |
| `PUT /api/v1/chat/<guid>` `{displayName}` | **CONFIRMED** | `httpRoutes.ts` (`path: ":guid"`, method `PUT` → `ChatRouter.update`); `update()` reads `body.displayName` (`chatRouter.ts:169-197`). Postman: `PUT {{host}}/api/v1/chat/<Chat GUID>`. |
| `POST /api/v1/chat/<guid>/participant/add` `{address}` | **CONFIRMED** | `httpRoutes.ts` line ~353 (`path: ":guid/participant/add"` → `ChatRouter.addParticipant`); `ChatValidator.toggleParticipantRules = {address: "required|string"}` (`chatValidator.ts:58-60`). (The server also exposes the shorter `POST/DELETE :guid/participant` as an alias for the same handlers — both work; the code's choice of the explicit `/add` path is fine and is also what Postman documents as a distinct request.) |
| `POST /api/v1/chat/<guid>/participant/remove` `{address}` | **CONFIRMED** | Same route table, `path: ":guid/participant/remove"` → `ChatRouter.removeParticipant`, same validator. |
| DM chat guid `iMessage;-;<handle>` | **CONFIRMED** | Not generated by the BlueBubbles server itself (it's an artifact of Apple's own `chat.guid` scheme in the Messages database), so there's no BlueBubbles source line to cite for the *format itself* — but `mautrix/imessage`'s independent BlueBubbles client constructs exactly this: `"iMessage;-;" + handle` (`imessage/bluebubbles/api.go:1389`), and BlueBubbles' own captured example response (see below) shows the parallel group form. |
| Group chat guid looks like `iMessage;+;chat…` | **CONFIRMED** | BlueBubbles' official Postman collection's saved live example response for `Chat / Get Chat Messages` contains a real chat object: `"guid": "iMessage;+;chat1234567890"`. This is an actual captured payload from BlueBubbles' own docs, not a guess. |
| `GET /api/v1/attachment/<guid>/download` | **CONFIRMED** | `httpRoutes.ts` (`prefix: "attachment"`, `path: ":guid/download"` → `AttachmentRouter.download`). Postman: `GET {{host}}/api/v1/attachment/<GUID>/download`. |
| Response envelope `{status, message, data}`, `status >= 400` ⇒ error | **CONFIRMED** | `responses/success.ts` builds `{status, message, data?, metadata?}`; the HTTP status code and the JSON `status` field are always set from the same value (`responses/errors.ts`), so `transport.py`'s belt-and-suspenders check (`response.status_code >= 400` and `payload["status"] >= 400`) is redundant but not wrong — no change needed. |

### Webhook: `new-message` payload shape

- **Envelope `{"type": ..., "data": {...}}`**: **CONFIRMED.** `WebhookService.dispatch(event: {type, data})` POSTs `event` directly as the JSON body (`services/webhookService/index.ts`).
- **`data` is the same shape as a serialized message** (i.e. `MessageResponse`), because `handleNewMessage` builds it via `MessageSerializer.serialize({message: newMessage, ...})` and passes that straight to `emitMessage(NEW_MESSAGE, ...)`, which dispatches the webhook (`server/index.ts:1499-1531`).
- **`guid`, `text`, `dateCreated` (ms epoch), `isFromMe`, `handle.address`, `chats[].guid`, `attachments[].guid`, `attachments[].mimeType`**: **CONFIRMED**, all present verbatim in `MessageResponse`/`HandleResponse`/`ChatResponse`/`AttachmentResponse` (`types.ts`) and in BlueBubbles' own captured example response (see below). `dateCreated` is `message.dateCreated.getTime()` — a JS ms-epoch integer (`MessageSerializer.ts:150`).
- **`threadOriginatorGuid`**: **CONFIRMED** present at the same name, sourced straight from the macOS Messages database's `message_thread_originator_guid` column with no server-side transformation (`MessageDecoder.ts:85`, `types.ts:67`).
- **`ROWID` as the message id field**: **CORRECTED.** There is no `ROWID` field in any serialized message response. The field is **`originalROWID`** (`types.ts:26`, and confirmed live in BlueBubbles' own captured Postman example: `"originalROWID": 132938`). `ROWID` is only the *internal SQLite column name*; it never appears at the top level of an API/webhook payload under that name. **This is a real bug**, and it affects `regulars/features/ingest.py::message_row_from_bluebubbles`, which read `payload["ROWID"]`. Fixed (see below). `regulars/bot/transport.py` does **not** use `ROWID` anywhere — it correctly uses `guid` as the message id — so it was not affected.
- **`associatedMessageType` / `associatedMessageGuid` (tapbacks)**: **PARTIALLY CORRECTED — the numeric codes are confirmed, but the wire *type* of the field is not a plain integer, and the two authoritative sources disagree with each other:**
  - The **numeric code table itself is CONFIRMED exactly as assumed** (2000–2005 = add a tapback, 3000–3005 = remove one), straight from BlueBubbles' own source: `packages/server/src/server/databases/transformers/MessageTypeTransformer.ts` — `love:2000, like:2001, dislike:2002, laugh:2003, emphasize:2004, question:2005`, and `-love:3000 … -question:3005` (removal = add + 1000). `1000`/`sticker` is a distinct, non-tapback associated-message type, matching the code's existing "sticker, edit, or other" fallthrough.
  - **But the field is not necessarily a raw integer.** The `Message` ORM entity declares `associatedMessageType` as `type: "text"` with a `MessageTypeTransformer` column transformer that turns the DB integer into a *word string* (`"laugh"`, `"-love"`, `"sticker"`, …) whenever TypeORM hydrates the entity normally (`entity/Message.ts:451-457`). And BlueBubbles' **own official Postman example response** shows exactly that: `"associatedMessageType": "laugh"` (a string, not `2003`).
  - However, the **live poller that actually feeds real-time webhooks** does not go through that entity-hydration path — `MessageDecoder.decode()` builds the `Message` object by hand from a raw SQL join and assigns `message.associatedMessageType = entry.message_associated_message_type` directly, bypassing the transformer, so in that path the value is the **raw integer** (`entity/decoders/MessageDecoder.ts:70`). This is the code path behind `handleNewMessage`, i.e. the actual `new-message` webhook.
  - I could not resolve this conflict with a live webhook capture (no such thing exists publicly), and I'm not going to present a guess as settled. **Practical conclusion, stated honestly as a risk, not a fact:** the webhook payload is most likely the raw integer per the current poller source, but BlueBubbles' own published example proves the word-string form is real and reachable via at least the REST message-list endpoints (and possibly older/different server versions or REST paths for the webhook). The previous code's `int(raw)` on a value like `"laugh"` throws `ValueError`, is swallowed as `None`, and **silently drops every tapback whose type ever arrives as a word string** — turning a tapback into nothing, not even a plain-text fallback. That is the one place where "not verified" would have produced a real, silent regression once BlueBubbles changes (or already has changed, per its own docs) the wire representation. **Fixed**: `transport.py` and `ingest.py` now accept both encodings (int, numeric string, or the documented word string) via the same lookup table BlueBubbles itself uses. This is the single most important variance this pass surfaced, precisely because it sits on the transport seam §9 calls out.
- **`associatedMessageGuid` prefix (`p:<part>/<guid>` / `bp:<guid>`)**: **CONFIRMED** — not documented in BlueBubbles' own docs (this is an Apple Messages-database quirk BlueBubbles passes through untouched: no prefix-stripping code exists anywhere in `bluebubbles-server`), but independently corroborated by `mautrix/imessage`'s own tapback parser handling the identical two prefixes the identical way (`imessage/tapback.go:55-58`: strip `"bp:"`; for `"p:<n>/<guid>"`, split on `/` and take the guid half). `regulars/bot/transport.py::_strip_part_prefix` and `regulars/features/ingest.py::_strip_part_prefix` already implement this correctly — no change needed.

### Not covered by any endpoint the code calls

Nothing else in `transport.py`/`ingest.py` makes an assumption not covered above.

## Bottom line

Of the assumptions flagged as unverified, all endpoint paths, request/response field names, the
auth mechanism, the response envelope, the tapback numeric code table, and the GUID-prefix
stripping logic were **confirmed correct** against BlueBubbles' real source and its own published
example payload. Two concrete, confirmed problems were found and fixed:

1. **`ROWID` → `originalROWID`** in `regulars/features/ingest.py::message_row_from_bluebubbles` — a
   straightforward wrong-field-name bug that would have made every real BlueBubbles import raise a
   `KeyError`.
2. **`associatedMessageType` may arrive as a word string, not only an integer** — both
   `transport.py` and `ingest.py` now parse either form, closing a silent-data-loss path on
   tapbacks that the original "not verified" comment specifically worried about.
