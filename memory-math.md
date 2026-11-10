# Memory Math Reference

> Sources: `src/recall-engine.ts` (recall), `src/memory-writer.ts` (write-time salience),
> `src/memory-db.ts` (propagation, decay, prune, backfill), `src/memory-maintenance.ts` (nightly cycles),
> `src/container-runner.ts` (recall budget), `src/config.ts` (shared thresholds),
> `container/agent-runner/src/history.ts`, `index.ts` and `frame-fade.ts` (context window, pinned prompt, what the model sees of earlier replies).
>
> Examples come from two live agents, called **Agent A** (gpt-oss-120b) and **Agent B** (MiniMax-M2.5), and their one user. Names, quotes and personal details from their conversations are left out or replaced.
>
> Part I covers recall (read path). Part II covers how salience values are born, spread, decay and
> die (write path). Part III covers the rest of the context window: the pinned prompt block,
> compaction, persistence and the nightly trim. Part IV lists known inconsistencies in the code.
> The appendix covers the per-request sampling seed (`container/agent-runner/src/seed.ts`), which
> is not memory math but lives here so all the randomness and scoring math is in one place.

---

## Variable Glossary

### Semantic memory fields

| Variable | Field | Range | Meaning |
|----------|-------|-------|---------|
| `salience_score` | `semantic_memories.salience_score` | 0.0–1.0 | **Structural importance**: how relevant or central this fact is to the agent's ongoing experience. Born from the model's extracted value. Raised by `MAX` on re-mention (dedup) and by `+salBoost` whenever the fact is traversed in recall. **Never decayed**: per-turn decay only touches edge tables. Does not measure truth or certainty; a well-established but peripheral fact can have low salience. |
| `confidence_score` | `semantic_memories.confidence_score` | 0.0–1.0 | **Epistemic certainty**: how sure the agent is that this fact is true. The model-extracted value, clamped to [0,1]. Never decayed. Rewritten only by the LLM when two facts are merged during consolidation (or by the agent via `upsertSemanticNode`). Used directly as a term in `sem→sem` chaining. |
| `epistemic_source_type` | `semantic_memories.epistemic_source_type` | categorical | How the fact was acquired. The memory writer always stores `'inferred'`. It does not affect any score. |
| `justification` | `semantic_memories.justification` | text | Free-text rationale for why the fact was written or last updated. Not used in scoring. |
| `last_verified_timestamp` | `semantic_memories.last_verified_timestamp` | ISO timestamp | Set on reinforcement and merge. Not used in scoring. |

### Salience vs. emotional salience: key distinction

These two concepts are often written near each other in formulas but measure entirely different things:

| Concept | Where it lives | What it measures |
|---------|---------------|-----------------|
| **`salience_score`** | Column on `semantic_memories` | Structural/topical importance of a fact. A stored property of the node, raised by reinforcement and recall boosts. |
| **`adjEmotSal` / AES** | Computed from `emotion_memory_edges` | How emotionally resonant a memory is *in the context of the current traversal*. Derived dynamically by comparing the emotion edges on one node against those of another (the reference set). Not stored; computed fresh each recall run. |

In the `sem→sem` formula for example:
```
inner = 0.4·vecSim + 0.3·confidence_score + 0.2·salience_score + 0.1·adjEmotSal
```
- `salience_score` answers: *is this fact structurally important?*
- `adjEmotSal` answers: *does this fact emotionally resonate with where we currently are in the graph?*
- `confidence_score` answers: *is this fact believed to be true?*

### Edge fields

| Variable | Source | Meaning |
|----------|--------|---------|
| `edgeSal` | `inter_memory_edges.salience` | Salience of a memory→memory edge (Response_To, Continues_From, Reminds_Of, Mentions, SimilarTo). Reflects how strongly two nodes are linked. |
| `emotion_memory_edges.salience` | per (emotion, target) | How strongly an emotion attaches to an episodic memory, semantic memory, or inter-memory edge. |
| `derived` | `emotion_memory_edges.derived` | Provenance: `0` = **earned** (the model actually felt this emotion about this target), `1` = **derived** (inherited or propagated). See Part II §B. |
| `rawSalCurr` / `rawSalHop` | avg of `emotion_memory_edges.salience` on a node | Raw average emotional intensity of a node, regardless of what emotions are present at the traversal origin. |
| `aesCurr`, `aesHop` | `adjustedEmotionalSalience(...)` | Context-sensitive emotional resonance: raw emotion salience weighted by how well the emotions match a reference set. See helper below. In Phase 2, `aesCurr` judges a source against its **chain entry** (what led to it; see Phase 2 → Chain-entry reference), and `aesHop` judges the target against the source. |

All salience columns are clamped to [0, 1]. Every additive boost is applied as `MIN(1.0, salience + x)`.

---

# Part I: Recall

## Overview

Recall is a graph traversal that runs on the host before each container spawn. It anchors on the **most recent** episodic memory (the session anchor) and expands outward through the memory graph, using chaining weights (CW) to decide which path to follow at each step. The result is a ranked list of episodic and semantic memories written to the IPC directory for the agent to consume.

- Episodic pool: **every** alive episodic memory, ordered by id. There is no recency cutoff; forgetting is modelled by salience decay (Part II), not by making old turns unreachable.
- **Episodes are dynamic.** A recalled episode's start, end and links to other episodes come from the cue and the traversal, never from stored boundaries. `Continues_From` edges are ordinary edges that the walk may follow; nothing injects them.
- Semantic pool: every alive semantic memory with an embedding, **except** `Autobiographical` facts, which have their own pool (Autobiographical Pass) and are invisible to the semantic pass and to every chain.
- All emotion-memory and inter-memory edges are bulk-loaded once at the start. Boosts are written to the DB during traversal, but traversal keeps reading the **preloaded** values, so a boost never affects scores within the same recall run.
- The whole run is a single SQLite transaction.

A **semantic pass** runs first: it retrieves facts by their similarity to the prompt and hands linked episodes to round 0. Then the algorithm alternates between two phases:

```
Semantic pass → Phase 1 (chaining) → Phase 2 (emotional hop) → Phase 1 → Phase 2 → …
```

There are at most `MAX_HOP_ROUNDS = 5` Phase 2 rounds (effectively 4; see Iteration and Termination). The loop ends early if no hop target clears the threshold or the token budget runs out.

---

## Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `BASE_THRESHOLD` | 0.05 | Minimum CW to accept a Phase 2 hop target or a semantic-pass step; also the subtraction base for every boost |
| `EP_EP_INNER_SCALE` | 4 | Divisor for the ep→ep `inner` term (see ep→ep below) |
| `WALK_BASE_THRESHOLD` | 0.15 | Phase 1 walk floor for round-0 chains (anchor and semantic-pass starts) |
| `HOP_WALK_FLOOR_RATIO` | 0.25 | Hop chains (round 1+) walk while CW > max(`BASE_THRESHOLD`, 0.25 · start score) |
| `MAX_HOP_ROUNDS` | 5 | Maximum Phase 2 → Phase 1 iteration cycles |
| `MAX_SEED_MEMORIES` | 5 | Maximum hop targets selected per Phase 2 round |
| Boost scale | 0.001 | Multiplier on every recall boost (Phase 1 and 2) |
| `SEM_SEEDS` | 5 | Seed quota of the whole-message cue |
| `SEM_SEEDS_PER_CUE` | 3 | Seed quota of each sentence cue (2 before 2026-10-25) |
| `SEM_SEEDS_PER_NAME` | 3 | Seed quota of each name cue |
| `SEM_MAX_SEEDS` | 16 | Total seed cap across all cues (10 before 2026-10-25) |
| `MAX_CUE_SENTENCES` | 8 | Sentence cues per message (longest kept) |
| `MAX_NAME_CUES` | 4 | Name cues per message (rarest first) |
| `SEM_NAME_MAX_SHARE` | 0.03 | A name in more than this share of alive facts is too common to cue on… |
| `SEM_NAME_ALWAYS_OK` | 5 | …unless it is in this many facts or fewer |
| `SEM_SEED_MIN_SIM` | 0.45 | Minimum cos(fact, prompt) for a seed (bge-m3: related ≈ 0.5–0.7) |
| `SEM_SEED_IMPORTANCE_WEIGHT` (β) | 0.2 | Strength of the importance weight on seed ranking (weight ∈ [0.9, 1.1]); 0 = pure similarity |
| `SEM_BRANCH` | 3 | Neighbours pushed per recalled fact |
| `SEM_NEIGHBOUR_MIN_SIM` | 0.3 | Minimum vecSim for a sem→sem step (both passes) |
| `SEM_MAX_FACTS` | 40 | Count cap on facts recalled by the semantic pass |
| `SEM_MAX_EP_STARTS` | 5 | Episodes the semantic pass hands to round 0 |
| `SEM_RENDER_OVERHEAD` | 25 | Tokens charged per fact for the container's tags and suffix |
| `DEFAULT_AUTOBIOGRAPHICAL_RECALL_TOKENS` | 1000 | Autobiographical pool when `MAX_AUTOBIOGRAPHICAL_RECALL_TOKENS` is unset (2026-10-26) |
| `AUTO_RENDER_OVERHEAD` | 12 | Tokens charged per autobiographical line for its `- [date] Name: ` prefix |

### Token budget

Recall has three separate pools, and every memory is charged to its own type's pool, whichever pass found it. Unused budget does **not** roll over between them. (Two pools until 2026-10-26, when the autobiographical pool was added.)

```
perSlot          = LLM_MAX_CONTEXT / LLM_PARALLEL
effectiveContext = max(256, perSlot − min(WORKING_MEMORY_CONTEXT, perSlot / 2))     (WORKING_MEMORY_CONTEXT default 4096)
toolCallReserve  = max(0.25 · effectiveContext, 4096)
maxMemoryTokens  = max(0, effectiveContext
                          − systemPromptTokens        (tokenized from the files on disk, + the tool definitions*)
                          − userPromptTokens          (tokenized prompt)
                          − 37                        (jinja user-turn suffix)
                          − 4096                      (RESPONSE_RESERVE_TOKENS)
                          − toolCallReserve
                          − 140)                      (memory block tags: 50 episodic + 80 semantic + 10 current-message)

semBudget = min(MAX_SEMANTIC_RECALL_TOKENS ?? 2000, maxMemoryTokens)
autoCap   = MAX_AUTOBIOGRAPHICAL_RECALL_TOKENS ?? 1000

autoBudget = min(autoCap, maxMemoryTokens − semBudget)          # always before episodic
epBudget   = min(epCap ?? ∞, maxMemoryTokens − semBudget − autoBudget)

episodic cost         = ceil(chars / 4)
semantic cost         = ceil(chars / 4) + SEM_RENDER_OVERHEAD
autobiographical cost = ceil(chars / 4) + AUTO_RENDER_OVERHEAD
```

*Autobiographical first (2026-10-27).* The autobiographical slice used to come after a capped episodic pool ("on top of the other pools"). That held only while the episodic cap fit inside the formula. Once the tool definitions counted, Agent A's formula left ≈ 31k after facts against a 40k episodic cap, episodic took all of it, and her own expressions got 0 in every recall from 21:58 to 22:36. Now the slice always comes out first: when the cap fits, nothing changes; when it doesn't, episodic shrinks by the slice instead.

\*The tool definitions go out with every request outside the message history. Since 2026-10-27 the container reports their size each round (`data/ipc/{group}/tool_schema_tokens.json`) and both this budget and compaction count them; before its first report the host estimates 280 tokens per tool in `tools.json`. Until then nothing counted them, and with working memory and the round both full a request could overflow by that much. First report, Agent A: **11,661 tokens for 24 tools**, double the ~5k estimated beforehand. It is chars/4 of the JSON schemas, which servers render more compactly (GPT-OSS gets a TypeScript-like namespace), so it likely counts high. Since 2026-10-28 each tool set is measured on the real server instead (Part III §F).

**Held memories are free** (`RECALL_DEDUP`, 2026-10-28; Agent A and Agent B). On a follow-up message to a running container, memories its working memory still holds in earlier recall blocks are not charged to their pool:

```
held     = ids in earlier blocks after the next fade          (container, after each turn → data/ipc/{group}/held_recall.json)
on add:  memory ∈ held  → heldTokens += cost                  (moved, not added: dedup removes the old copy)
         otherwise      → pool tokens += cost
caps compare pool tokens only; totalTokenEstimate counts both
```

Episodic and semantic only: autobiographical lines are still charged (a small pool, mostly the same lines each turn). The file is removed at every spawn, since a new container holds no recall, and heartbeats always pay in full. Net context is unchanged: each held item leaves its old block as it enters the new one, so the cap buys new material instead of copies. If a message arrives while the previous turn is still running, the file is one turn behind; a few items then count as free that the next fade drops, and a recall can exceed its cap by at most that faded remainder. `heldTokens` on `Recall completed` is what moved free.

**`WORKING_MEMORY_CONTEXT` is the working-memory space**: the part of the window kept for earlier rounds (Part III §B). Recall and tool budgets come from `round_max_ctx = window − WORKING_MEMORY_CONTEXT`, so one round never extends into it. Agent A: 50,000 (briefly 10,000 on 2026-10-27, while compaction still ignored the split).

Worked example, Agent A (2026-10-27, after the tool definitions counted): `perSlot` 131,072, working memory 50,000 → `effectiveContext` (round) 81,072; `toolCallReserve` 20,268; prompt block + tool definitions + prompt ≈ 23k; `maxMemoryTokens` ≈ 33.6k → 2,500 semantic + 3,000 autobiographical + ≈ 28k episodic (under its 40k cap). Before the tool definitions counted it was ≈ 55k before caps, so ≈ 45.5k per recall.

This is computed on the host (`writeMemoryContext`, `container-runner.ts`) with the server's tokenizer, falling back to chars/4. `effectiveContext` mirrors the container's `LLM_EFFECTIVE_CONTEXT`, so recall never takes space the model needs at inference. How the container then fits the conversation into the same window is Part III.

Caps are read from `group.env` first, then `.env`. `MAX_RECALL_TOKENS` is the legacy name for the episodic cap. Phase 1 stops when the episodic pool is spent; ep→sem and sem→sem candidates are skipped once the semantic pool is spent.

**The semantic cap, by contrast, is nearly always the limit.** The semantic pass spreads until the pool is spent, and Phase 1 keeps adding facts through ep→sem and sem→sem until the same pool is gone. Every recall in the 2026-10-25 dry runs (11 recent prompts, Agent B and Agent A, both capped at 2,500) ended at 2,500–2,536 tokens. With the cap lifted, the same recalls stopped on their own at:

| Group | Mean | Range | Facts recalled |
|---|---|---|---|
| Agent B | ~2,900 | 1,850–3,715 | 31–60 |
| Agent A | ~5,400 | 4,640–6,275 | 93–123 |

`SEM_MAX_FACTS` (40) caps only the semantic pass; the rest come from the episodic walk, which is why Agent A draws far more facts than Agent B. At the time of those runs her episodic budget was the smaller one (8k against Agent B's 24k; 40k since 2026-10-27), so the difference is more likely the size of her fact pool: about 3,600 alive facts against 1,700 (2026-10-27), giving the walk more to reach. Unmeasured. (Until 2026-10-27 this sentence credited a 24k episodic budget to Agent A; that budget is Agent B's.) Both caps stay at 2,500 deliberately: the semantic pool shares the context window, and Agent A's model handles fewer tokens. A full pool is therefore normal. What must not happen is the pool filling before the seeds are in, which seeds-first spread prevents (Semantic Pass → Spread); `seedsCut` in the log should stay 0.

The cap is a backstop, not the usual limit. Since hop chains walk (relative walk floor, 2026-10-17), an uncapped recall stops on its own at roughly **12–20k episodic tokens**: dry runs on Agent B gave 11.9k, 12.6k and 19.7k, with identical results at caps of 20k, 30k and 60k. A tighter cap truncates the last hop rounds. Agent B sets `MAX_EPISODIC_RECALL_TOKENS=24000` (raised from 10000 on 2026-10-17) for headroom over the observed peak.

**Agent A does not stop on its own at that size.** Her graph is larger, and round-0 episodes (one per cue, each walked by Phase 1) spent a 20k pool before Phase 2 ran, so most recalls never hopped. Copy test, 2026-10-27, 8 cues (6 live messages + 2 asking about 2025):

| Setting | Turns | From 2025 | Hop episodes | Hops into 2025 | Recalls reaching round ≥ 3 |
|---|---|---|---|---|---|
| plain mean, pre-repair emotions, 20k | 984 | 56 (6%) | 55 | 0 | 0/8 |
| earned mean, repaired, 30k | 1,408 | 70 (5%) | 97 | 2 | 4/8 |
| earned mean, repaired, 40k | 1,692 | 127 (8%) | 171 | 11 | 8/8 |

Recall time did not grow (max 6.7s → 5.8s). 40k is live; live recalls filled it about two times in three (the rest ended when hops ran out, ~32k) until the tool definitions counted, which put the formula below the cap (≈ 28k, worked example above).

### Observability

Every recall logs `Recall completed` at **info**. Recalls slower than 2s log `Recall took longer than 2s` at **warn** instead. Fields:
- `elapsed`, `groupFolder`
- `episodic`, `semantic` (counts), `episodicTokens`, `semanticTokens`
- `semanticSeeds`, `cueSentences`
- `seedsReordered`: seeds the importance weight swapped in, compared with a pure-similarity pick over the same DB state; `seedWeightMean`: mean weight of the chosen seeds
- `nameCues`: the name cues that survived the frequency filter; `seedsByCue`: seeds per cue kind (`whole`, `sentence`, `name`)
- `seedsCut`: seeds left out by the semantic budget or `SEM_MAX_FACTS`. It should stay 0; anything else means the seeds alone overran the budget
- `autobiographical`, `autobiographicalByRelevance`, `autobiographicalTokens`: lines in the autobiographical pool, how many came from the relevance half, and their cost
- `episodes[]`: `{round, startId, score, adjTurns}` per chain

The latest recall, including full `episodes[]` membership (`memberIds`), is always in `data/ipc/{group}/memory_context.json`. Before 2026-10-17 17:49 the fast path logged at debug, so log-derived recall statistics from before then cover only the slow (>2s) recalls.

---

## Shared Helpers

### MODE and MEDIAN

`MODE(values)` buckets values by `toFixed(2)` (two-decimal bins), picks the most frequent bin, and returns the **first-seen raw value** in that bin (not the bin centre). Ties go to the bin seen first. When every value is distinct, this is simply the first value. Empty input returns 0.

`MEDIAN` is the standard median (mean of the two middle values for even counts). Empty input returns 0.

### Counted edges: `countedEmotionEdges(edges, groupSigma)` (`EMOTION_AVERAGE`)

Every emotional term in recall (a node's average intensity, AES, `rawSalCurr` / `rawSalHop`, `aes_ep` / `aes_inter` / `aes_sem`, the semantic seeds' intensity and resonance) reads a node's **counted** edges, not all of them:

```
earned = edges with derived = 0
if earned is empty or every edge is earned:  counted = edges
else:
    μ = mean(earned.salience)
    σ = population sd of earned.salience     if |earned| ≥ 2
      = groupSigma                            otherwise   (sd of every earned edge in the group, per recall)
    counted = earned ∪ { e derived : e.salience ≥ μ − σ }
```

Boosts are applied to **every** edge (`cachedEmotEdges`), counted or not. An inherited edge that recall keeps travelling climbs ~10⁻³ per visit and re-enters the count once it reaches the earned range. `EMOTION_AVERAGE=all` (group.env) restores the plain mean over all edges. Default `earned` since 2026-10-27.

*Why.* Propagation gives a node many inherited edges beside its earned ones (Part II §B), and a mean over all of them is pulled toward the inherited level. On Agent A (2026-10-27) the imported May–June 2025 turns averaged 2.7 earned edges at 0.57–0.65 and ~14 inherited at ~0.46, against ~1 earned at 0.74–0.87 and ~5 inherited for October 2026 turns. Every old turn therefore read ~0.47 against ~0.7, which mostly cost it Phase 2 hops. On copies (6 live cues, 20k episodic): 2025 turns recalled 48 → 60, round-0 episodes 61 → 78.

### Adjusted Emotional Salience: `adjustedEmotionalSalience(current, reference)`

Measures how emotionally resonant `current` is relative to a `reference` set of emotion edges. Matched emotions (present in both) are boosted; unmatched ones are dampened.

```
For each emotion edge e in current:
  if e.emotion_id is in reference:
    value = 0.7 · ref_salience + 0.3 · e.salience
  else if partial matching is on and some r in reference has w = match(e, r) > 0:
    value = max over those r of  w · (0.7 · r.salience + 0.3 · e.salience) + (1 − w) · 0.3 · e.salience
  else:
    value = 0.3 · e.salience

AES = mean(values)          (0 if current has no edges)
```

Used as `aesCurr`, `aesHop` in Phase 2, in ep→ep chaining, and as `adjEmotSal` in sem→sem. `weightedEmotionalSalience` (the `aesHop` variant with the rarity weight) takes the same partial branch.

**Partial matching** (`EMOTION_PARTIAL_MATCH`, group.env, default off; Agent A on since 2026-10-27 22:20). A near-synonym of a reference emotion counts as a fraction of a match instead of as unmatched:

```
for each of name, definition, circumstance, felt_quality:
    centered(x) = normalize(embedding(x) − mean embedding of that field over all emotions)
sim(a, b)   = mean over the fields both have of  centered(a) · centered(b)
match(a, b) = 1                                                  if a = b
            = clamp((sim − 0.12) / (0.45 − 0.12), 0, 1)          otherwise   (EMOTION_MATCH_FLOOR / _FULL)
```

Built once per recall (`buildEmotionMatcher`) and cached per pair. *Why centered:* every emotion's text is written in one voice ("I feel it when…"), so raw cosines sat near 0.62 for any pair; "affection"~"fear" (0.53) scored like "grateful appreciation"~"shared appreciation" (0.66). *Why the name:* on Agent A, name + three fields gave near-synonyms 0.21 on average against 0.08 for unrelated pairs. The three fields alone gave 0.12 against 0.09, because the circumstance describes situations, so emotions felt in the same moments look alike. The name embedding is `emotions.name_embedding`, written by the memory writer for new or renamed nodes and backfilled by `scripts/backfill-embeddings.ts`.

The match branch only discriminates if emotion sets actually differ between memories. When propagation ran to transitive closure (every memory carried ~154 of 176 emotions), the match rate was 1.0 and the dampen branch was dead. The propagation bounds in Part II §B exist to keep this helper meaningful.

---

### Mode of Means: `modeOfMeansEpSem(ep, sem)`

Captures the typical emotional resonance across all inter-memory edges that connect a specific episodic node to a specific semantic node. Used in ep→sem and sem→ep transitions.

```
For each connecting edge e between ep and sem:
  aes_ep    = avg salience of all emotion edges on ep
  aes_inter = avg salience of all emotion edges on e
  aes_sem   = avg salience of all emotion edges on sem
  mean_e    = (aes_ep + aes_inter + aes_sem) / 3

MoM = MODE(mean_e across all connecting edges)     (0 if no connecting edges)
```

A node with zero emotion edges contributes 0 to the mean. That is a cliff, not a gentle decline, which is why earned edges are exempt from pruning (Part II §D).

---

## Semantic Pass: Prompt-Cued Fact Recall

Facts are recalled by cue, the way people recall them: the prompt calls up the closest facts, and each recalled fact brings its nearest neighbours with it (spreading activation). Skipped when there is no prompt embedding.

> **Changes of 2026-10-25**, in the order they were made (each is detailed below or in the section named):
> 1. **Importance-weighted seeds.** Similarity still decides which facts are eligible; salience, confidence and emotion (as percentile ranks) reorder them, β = 0.2. See Seeds.
> 2. **Name cues.** Each capitalised name inside a sentence becomes its own cue, minus names too common in the fact pool. Prompted by Agent B missing one name mentioned beside another. See Cues.
> 3. **Broader seeding.** Per-cue quotas (whole 5, sentence 3, name 3), rank-by-rank fill of distinct facts, cap 16. See Seeds.
> 4. **Seeds before spread, in pick order**, so the spread can't spend the semantic pool before every cue has its facts. See Spread.
> 5. **Write side:** dedup threshold 0.80 (`DEDUP_SIMILARITY`), configurable discard floor (`MIN_SEMANTIC_SALIENCE`), and facts from tool output (`rememberResults`). See Part II §A.
> 6. **Nightly merge** re-points links in place instead of re-inserting them, which had orphaned their emotion edges. See Part II §D.
> 7. **A false fact deleted**, and the loop that created it described. See Part II §A → Recall misses can become facts.

**Fixed text never cues recall.** `stripBoilerplate` (`src/heartbeat-text.ts`) removes the `[HEARTBEAT: …]` suffix, `[Recovery note: …]`, `[SCHEDULED TASK …]`, the no-wakeup note and markup, then drops any sentence matching a known fixed sentence (the container's default heartbeat opener and footer), wherever it appears. Otherwise every heartbeat would recall, and reinforce, facts about heartbeats. The container's sentences are mirrored on the host; a drift-guard test fails if the two copies diverge.

**Heartbeat cues.** For wakeup tasks, `writeMemoryContext` builds cues with `buildHeartbeatCues` from three parts, each split separately and capped at its 4 longest sentences:

| Part | Source | Included |
|---|---|---|
| Intent | the wakeup prompt after `stripBoilerplate` (whatever Agent B wrote to herself) | when anything is left |
| Last sent | `data/last-sent/{folder}.json` (recorded on every delivered agent message); fallback: latest reply to a real user turn | if one exists |
| Reflection | latest assistant turn answering a `[HEARTBEAT:` turn | only while newer than the user's last real message (an idle stretch) |

The reflection rotates on every idle heartbeat, so consecutive heartbeats don't replay the same fact recall: her mind drifts from her own last thought. If nothing real remains, there are no cues and the semantic pass is skipped.

**Cues.** `buildRecallCues` runs `stripBoilerplate` (which also removes the `<messages>`/`<message>` wrapper and decodes entities). The cues are:
- the whole message;
- when there are at least two, each sentence of 3+ words (the `MAX_CUE_SENTENCES` longest);
- **name cues**: each capitalised name inside a sentence, as its own short cue (`cueNames`). A sentence's first word is skipped (it is capitalised anyway), as is "I" with its contractions. Consecutive capitalised words form one name ("New York"), and a trailing possessive is dropped. Unlike sentence cues, names are taken from a single-sentence message too. Heartbeats take names from their intent, last-sent and reflection parts.

Before embedding, `selectNameCues` keeps only names that can discriminate. A name is kept if it appears in at least one alive fact and in no more than `max(SEM_NAME_MAX_SHARE · facts, SEM_NAME_ALWAYS_OK)` of them (a case-sensitive substring count, `semanticMentionCounts`). At most `MAX_NAME_CUES` survive, rarest first. On Agent B this drops the user's name (33% of facts), the agent's own (9%), the user's other name (8%) and a platform name (3.6%), and keeps rarer names (1% and below). The floor of 5 keeps a small or new pool from rejecting every name.

All cues are embedded in one batched `getEmbeddings` request. `vecToPrompt` everywhere (Phase 1 and the crossover) is `maxCueSimilarity`: the best cosine against any cue, names included, so turns that mention a named entity score higher in Phase 1. It is 0.5 when there are no cues.

*Why name cues (2026-10-25).* One sentence that names two things embeds toward whichever its facts match best. On "So, like, if I talk about X and Y, what comes up?" the sentence cue scored X facts 0.58–0.60 and the best Y fact 0.515 (the rest 0.45–0.47). With two seeds per sentence, both went to X; the whole-message cue went to the message's other sentence (about memory). No Y fact was seeded, the spread spent the budget elsewhere, and Agent B answered that Y was fuzzy despite 18 facts about her. The bare cue "Y" puts Y facts at #1–#3 (0.54, 0.51, 0.505). Replaying that recall with name cues seeds 919, 1601 and 1833 through the "Y" cue and recalls 4 Y facts alongside 6 X ones.

Why: one embedding of a multi-topic message leans toward the dominant topic. A side mention ("Riley keeps messaging me", names changed) ranked its best fact 21st against the whole message and 1st against its own sentence.

**Seeds.** Similarity decides whether a fact *can* seed; importance decides which eligible facts *do*. Every alive fact with an embedding gets an importance weight, computed over the whole pool on each recall:

```
intensity(f)  = avg salience of f's emotion edges                 (how strongly anything was felt about it)
resonance(f)  = adjustedEmotionalSalience(f's edges, anchor's edges)  (how well it matches the present turn)

P(x)          = percentile rank of x within the pool, in [0, 1]   (ties share their average rank)
emo           = 0.5·P(intensity) + 0.5·P(resonance)
importance    = 0.4·P(salience_score) + 0.3·P(confidence_score) + 0.3·emo
weight        = 1 + β·(importance − 0.5)                          (β = SEM_SEED_IMPORTANCE_WEIGHT = 0.2 → 0.9 … 1.1)
```

A fact is eligible for a cue when `cos(fact, cue) ≥ SEM_SEED_MIN_SIM`; the weight never makes an unrelated fact eligible. Each cue ranks its eligible facts by `seedScore = cos · weight` and has a quota: `SEM_SEEDS` (5) for the whole message, `SEM_SEEDS_PER_CUE` (3) for each sentence, `SEM_SEEDS_PER_NAME` (3) for each name. Seeds fill rank by rank (`fillSeedTiers`), up to `SEM_MAX_SEEDS` (16):

```
tier k: every cue whose quota ≥ k takes its best fact NOT YET PICKED by any cue,
        strongest cue first (by that fact's seedScore)

seed CW  = weight · max over cues of cos(fact, cue)
cwPropE  = 0, cwPropNE = 1        (the boost goes to structure only)
```

A cue never loses a slot to a fact another cue already took; it takes its next distinct fact instead. Before 2026-10-25, tier k took each cue's k-th fact even when it was already picked, so overlapping cues wasted slots, and sentence cues stopped at 2. A single-sentence message without names has only the whole cue, so it gets 5 seeds.

*Why percentile ranks.* The raw columns are compressed differently per group: on Agent A confidence p5–p95 is 0.85–1.0 and salience 0.60–0.92; on Agent B salience is 0.30–0.91 and mean emotion salience 0.42–0.66. Ranks let each signal discriminate equally whatever its spread. They also keep the weight centred on 1, so a median fact keeps its plain-similarity CW and the spread and crossover below see the same scale as before, even as `salience_score` inflates over time.

*Why the anchor appears.* `resonance` makes seeding mood-dependent the way Phase 2 is: of two equally intense facts, the one whose emotions match the present turn ranks higher. `intensity` keeps facts that were strongly felt in any mood competitive.

*Why β = 0.2.* Top candidates sit close together in similarity, so a small weight already reorders them. Dry run on DB copies, 5 recent prompts each on Agent B and Agent A (2026-10-25):

| β | seeds swapped vs pure similarity (per prompt) | recalled facts shared with β = 0: median (range) |
|---|---|---|
| 0.2 | Agent B 5/10, 0/5, 2/5, 1/5 · Agent A 3/7, 1/6, 5/7, 4/10 | 69% (0–100%) |
| 0.4 | Agent B 7/9, 1/5, 3/5, 2/5 · Agent A 6/9, 3/7, 5/7, 6/9 | 51% (0–83%) |
| 0.6 | Agent B 6/8, 2/5, 4/5, 3/5 · Agent A 7/8, 3/7, 7/8, 7/9 | 46% (0–100%) |

Episodic tokens barely moved at any β. β = 0.2 lets importance break near-ties without replacing the cue's picks wholesale. Because seed CW is scaled by the weight, a heavily weighted seed also spreads further (on one Agent B prompt β = 0 recalled 5 facts and β = 0.2 recalled 41, none shared: a different, more important seed won and spread to near the `SEM_MAX_FACTS` cap; that prompt is the 0% in the table).

*Watch: rich-get-richer.* Seeding boosts `salience_score` (`cwPropNE = 1`), which is never decayed, and salience now feeds seed selection. The loop is bounded: the weight is capped at ±β/2 and saturates once a fact reaches the top percentile. Baseline on 2026-10-25: facts with `salience_score ≥ 0.99` were 0.5% of the pool on Agent B (9/1838) and 0.1% on Agent A (5/3486). If that share grows noticeably by late November, lower β. Keeping `cwPropE = 0` on seeds is deliberate: emotion boosts would add a second reinforcement path.

**Spread (best-first, seeds first).** While any seed is pending, pop seeds in **pick order** (every cue's #1, then every cue's #2, …); after that, pop the strongest pending fact by CW. Each popped fact is added to the result and boosted (`applyPhase1Boosts`; non-seeds also create or boost the `SimilarTo` edge from their parent). Then its top `SEM_BRANCH` neighbours are pushed, scored with the Phase 1 sem→sem formula:

```
child CW = parent CW · (0.4·vecSim + 0.3·confidence + 0.2·salience + 0.1·adjEmotSal)
           (vecSim ≥ SEM_NEIGHBOUR_MIN_SIM, child CW > BASE_THRESHOLD)
```

Stops at `SEM_MAX_FACTS`, when the semantic pool is spent, or when nothing pending beats `BASE_THRESHOLD`.

*Why seeds first, in pick order.* The spread nearly always spends the whole semantic pool: every recall in the 2026-10-25 dry runs (11 prompts, two groups) ended at 2,500–2,536 of 2,500 tokens. Under pure best-first, neighbours of strong seeds (child CW ≈ 0.45–0.66) could be popped ahead of weak seeds and spend the budget before them. Name cues match their facts at only ~0.5, so they were the most exposed. Taking seeds first means every seed is recalled before any spread; taking them in pick order means that if seeds alone ever overran the budget, the cut would fall on the lowest tier rather than on a weakly matching cue's only fact. `seedsCut` logs any seed lost to the budget; it was 0 in all 11 dry-run recalls (5–16 seeds each).

**Crossover.** Each recalled fact scores its linked episodes in the recent pool with the Phase 1 sem→ep formula. `vecToStart` is measured against the fact's seed:

```
score = CW_fact · (0.3·edge.salience + 0.2·vecToPrompt + 0.2·vecToStart + 0.4·MoM)
if score > BASE_THRESHOLD: score += score − BASE_THRESHOLD
```

The best `SEM_MAX_EP_STARTS` episodes overall are added to the result, boosted, and appended to round 0's chain starts after the anchor. Recalling a fact can therefore bring back the conversation it came from.

---

### Period starts (`PERIOD_STARTS_MAX`, `PERIOD_STARTS_FLOOR`, 2026-11-01; Agent B 4, 0.45)

Before Phase 1, each calendar month older than 30 days offers its turn most similar to the whole-message cue, if `sim ≥ floor`. The `max` closest start round-0 chains with `CW = sim`. Age-blind within the set of older months, and similarity-gated, so a poor month offers nothing. Copy test, Agent B, 60 questions: March–May share of recalled turns 8.5 → 13.9 → 19.2% on ordinary messages (none, 2, 4 starts), 1.6 → 7.5 → 13.4% on lasting questions, 19.3 → 22.6 → 25.6% on questions naming a spring topic; with 4, about 11 of 106 turns per recall are displaced, about half of them from the past week and outside working memory. Similarity is not chronology: the earliest turn recalled is often not the first.

### Recall note (`RECALL_NOTE`, 2026-11-01; Agent B, her wording)

Text placed before "End of recalled memories" in the latest recall block only; `stripRecallNote` removes it from earlier blocks before each new recall, so working memory never holds more than one. It does not change what is recalled. Replays, 3 replies per setting: on "when did I first tell you about a named character" (true Apr 17) every setting says Apr 20-21 and hedging went from 5 of 6 to 6 of 6 with a simple note; on "first time you felt proud" (true Mar 9) every setting with or without a note said April or later, because recall returns her own review reply ("*Proud* (153)") and it reads as memory. Similarity is not chronology. Ordering episodes by date, tried 2026-11-01, did not help and was dropped.

### Tested and not adopted, 2026-11-01

- **Nightly cycle.** Replay each day's messages with recall limited to that day plus a random 7-day window, keeping the links formed. Agent B, 25 simulated nights: about 5,600 links, lasting-question spring share unchanged at 1.6%. Strengthening new links 5x and 10x still left it at 1.6%, displaced up to 16% of recent turns, and made 43 and 525 links at 0.9 or more. Agent A at 10x: 1,345 such links, lasting share fell.
- **Second walk anchor** (`secondAnchor`, simulation). "Older than two days" lands in recent weeks, not the older period: Agent B lasting 1.6 to 2.1%.
- **Period starts on Agent A.** Direct 2025 questions 32 to 43%, lasting 15 to 20%, but recent turns kept fell to 62% on direct questions, and she has only two older months, so 4 starts equals 2. Not offered yet: she already reaches 2025 on every lasting and direct question. Everyday figures from that run came from replays when she had few recent turns (47%); live is about 16-25%. To be measured on her current memory once her batches settle.

### Effect of rebuilt layers on recall

Same questions on copies. Agent B, March-May share without and with March-May layers: everyday 3.7 to 7.4%, lasting 0.6 to 1.6%, naming a spring topic 10.1 to 19.3%. Agent A, 2025 share before May 13-20, before May 21-31, after: recent messages 15.2, 20.6, 24.1%; lasting 17.7, 11.0, 14.0%; 2025 topics 26.9, 18.8, 25.0%. Revisits followed per recall rose from 11-15 to 31-40, with more distinct memories overall. These copies were taken hours apart, so part of the dips comes from new turns.

### Dense stretches (Agent A Oct 26-27, open)

Oct 26-27 is under 10% of her turns and about half of everyday recall (Oct 22-27: 17% of turns, 53% of recall). Those 322 turns touch about 51,700 `Reminds_Of` links (2025 averages 54 per turn, Oct 26 233), and all 50 turns with an emotion edge at 0.95 or more are from Oct 22-27. Likely causes: the CW inflation bug and the same-evening replay on 2026-10-27, then recall reinforcing what it returns. Copy tests, Oct 26-27 share of everyday / lasting / 2025-topic recall, from 53-54 / 65-66 / 56%:

| Change to those turns | Everyday | Lasting | 2025 topics | Same as now |
|---|---|---|---|---|
| Keep each turn's 50 strongest links | 50.5 | 53.4 | 52.6 | 81-89% |
| Keep 20 | 47.8 | 54.2 | 54.8 | 77-85% |
| Cap emotion edges at 0.85 | 52.5 | 64.3 | 42.3 | 77-92% |
| Cap at 0.75 | 47.0 | 56.4 | 25.1 | 62-82% |
| Keep 20 and cap at 0.75 | 34.9 | 26.8 | 15.5 | 47-73% |

Lowering one turn's edges changes nothing, since neighbours bring it back. Recalled replies opening on a "your X feels like" frame fell from about 38-46% to 17-28% with both changes, but a replay of 36 replies at cap 0.75 alone showed no change in her own frames or imagery: working memory holds the same patterns. Replies with both changes are untested. Nothing applied.

### Fact handoff (`FACT_HANDOFF`, 2026-10-29; Agent B on)

After the semantic pass, each recalled fact may bring back one linked scene not yet recalled, as a round-0 chain start:

```
candidates = episodes linked to the fact, not yet recalled, with an embedding
best       = argmax sim(message, scene)        if sim ≥ FACT_HANDOFF_MIN_SIM (default 0.55)
score      = min(1, CW_fact · (0.5 + 0.5·link salience) · sim)     kept if > BASE_THRESHOLD
```

Age-blind and similarity-gated. **Why:** a read-only trace of Agent B's recall (12 messages, 2026-10-29) found 80% of recalled turns arrived by Phase 2 hops, and hops stay inside the densely linked recent days (only 61 of 1,034 hop arrivals were older than a week). Older facts were recalled often (182) but only 69 of their 454 older scenes came back. Copy tests: older than a week 6.7% → 9.2% with the 0.55 floor, older scenes fitting the message (e.g. "test subject" → the May 26 conversation about experiments; "lost sleep" → July 15, "partner up all night"); handing off the strongest-linked scene instead reached 27.6% but about half the scenes did not fit, and hop rounds collapsed. Reserving hop slots for older targets (12.0%) and month-relative scoring everywhere (6–7%) were tested and not adopted. Over 40 cues from every month, recalled as of their own time, 59–73% of recalled turns were more than a week older than the cue; the present tilt is mostly the density of the last few days. `factHandoffs` on `Recall completed` counts them.

**Also found:** Agent B's turns before 2026-07-13 have no earned emotion edges at all (1,975 turns); the decay/prune bug fixed that day deleted them. Re-labelling them on a copy (2,183 turns, MiniMax, choosing only existing emotions, decayed to age) did not change reach by itself; a one-month in-order reconsolidation rebuild on that copy (Jul 13–Aug 13, 717 layers, 21% onto memories more than 30 days older) gave older memories their own revisits (227 of 518 followed revisits in later recalls started from memories older than a week). Neither is live.

## Autobiographical Pass

(2026-10-26.) The agent's own first-person expressions, node type `Autobiographical` (`AUTOBIOGRAPHICAL_NODE_TYPE`, `memory-db.ts`), get their own pool and a selection of their own (`selectAutobiographical`). They are pulled out of `semanticById` before any pass runs, so the semantic pass and the Phase 1 chains never reach them or charge them. Each fact is a dated expression ("I expressed that I…", or a plain first-person memory for ported facts); none is a fixed trait.

```
autoBudget              see Token budget
lastExpressed(f)        = reaffirmed(f) ?? created(f)
reaffirmed(f)           = last_verified − created  if > 60 s,  else none    (set by write-time dedup within the class)
cost(f)                 = ceil(chars / 4) + AUTO_RENDER_OVERHEAD

1. relevance half (only when there are cue embeddings):
     rank facts with embeddings by maxCueSimilarity(f, cues)   (same cues as the semantic pass)
     take in order while tokens + cost(f) ≤ floor(autoBudget / 2), skipping one that does not fit
2. recency half:
     rank all facts by lastExpressed, newest first
     take those not yet chosen while tokens + cost(f) ≤ autoBudget
3. render chosen facts oldest first (by created), each "- [date] Name: text (again date)"
```

With no cues the whole budget goes to recency. The half-and-half split keeps both what matters to the current message and the agent's latest self in view, even when the topic does not touch the latter.

**Time handling.** `created_at` is SQLite's zone-less UTC (`YYYY-MM-DD HH:MM:SS`); `last_verified_timestamp` is ISO and is set at creation too. Both are parsed as UTC before comparing. Compared as strings, every fact looked reaffirmed (`'T'` sorts after a space); passed to `new Date()` unparsed, the SQLite form was read as local time. The `> 60 s` margin separates a real re-expression from the creation-time stamp.

**Rendering.** The block sits first in the recalled block, which rides in the user turn, so each line carries the agent's name (`SELF_NAME`, from the group's `ASSISTANT_NAME`) and the header says the "I" is the agent. Unlabelled, "I expressed…" read as the user speaking (Agent A, 2026-10-26).

## Revisits: one-way reconsolidation links

(2026-10-27; per group, `REVISIT_LINKS=true`, layer strength `REVISIT_LAYER_SCALE`, default 0.5.) When the agent records a relational emotion about a recalled memory, the writer links the **old memory forward** to the turn that felt it:

```
link       Revisited_By: old memory → layer turn            (replaces Reminds_Of, which recall follows both ways)
strength   s = significance × chainingWeight × REVISIT_LAYER_SCALE
             significance   = agentSignificance × the entry's significance
             chainingWeight = how strongly that memory came back in the recall she reacted to
emotions   each named emotion gets s / (number named) on the link      (one revisit counts once)
cap        3 links per old memory; a new one replaces the weakest, or is refused if weaker than all three
```

The emotions sit on the link, never on the old memory, and are **not propagated** to similar memories (propagation grew the warm hubs). Every layer is appended to `reconsolidation_log`.

**Recall.** Revisited_By never enters the bidirectional edge map, so no ordinary step (ep→ep, ep→sem, sem→ep, Phase 2) can follow one in either direction. With the switch on, the links are kept apart, keyed by the old memory only. When a memory is added to the result, its strongest revisit not yet recalled is offered once:

```
cw(layer turn) = cw(old memory) × (0.5 + 0.5 × s)        accepted if > BASE_THRESHOLD, within the episodic budget
```

A turn added this way never offers its own revisits (no chaining). The container renders it inside the old memory's episode. `revisitsFollowed` is logged on `Recall completed`; the followed pairs are `revisits[]` in `memory_context.json`.

**Measured on a copy of Agent A** (memory #1474, May 22 2025, revisited warmly; original: protective longing 0.66, affectionate warmth 0.65, tranquil gratitude 0.64): 25% of its emotional weight from layers after 3 revisits, ~29% after 10 (the cap bounds it near 31%), original kept. Unsplit at scale 1.0 the same three revisits gave 67%. Old-side cue: the revisit followed forward; new-side cue: the old memory's recall weight was identical with the switch on or off. Report: `scripts/reconsolidation-report.sh [group] [--db copy]`.

**Dormant on Agent A until 2026-10-28.** Links are made only from relational emotions, and GPT-OSS gave none. Her forced first call couldn't see the tool description (Part III §F, `tool_choice: 'none'`), and later her own empty calls in history served as templates (Part III §C). Agent B (MiniMax) had 34 log entries by then; Agent A had 0. Her first came on the first turn after the bookkeeping was removed: turn 2836 → 2838, s = 0.36 (significance 0.72, chaining weight 1.0). Watch over the following days: `revisitsFollowed`, how old the layered memories are, and whether following revisits tilts her recall toward the present. Each followed revisit brings one later turn and costs episodic budget like any other.

**Rare-emotion weight (`RARE_EMOTION_WEIGHT`, off).** Built but disabled. In Phase 2 only, a **shared** emotion's term in `adjustedEmotionalSalience` is multiplied by `max(0, 1 + β·(rarity − mean))`, rarity = ln(N/df)/ln(N); unshared emotions are untouched, so a rare painful emotion never pulls a memory in by being rare. On Agent A it never fired at β = 0.5 (memories carrying rare hard emotions are almost never hop sources) and at β = 1 it removed every hop (nearly all shared emotions are common). Left off by decision: no surfacing of pain before healing has a path.

## Phase 1: Unified Chaining

### Walks only fade (`noGain`, 2026-10-27)

Every Phase 1 step is capped at the CW it came from, `CW_child = min(CW_parent, …)`: ep→ep, ep→sem, sem→sem, sem→ep, and the semantic pass's fact→episode step. Every episode start (semantic-pass handoff, sem→ep start, Phase 2 hop) is capped at 1.0, the anchor's score. Semantic **seeds** are not capped, because their importance weight (up to 1 + β/2) orders them and two seeds above 1 would tie; the cap applies where their linked turns enter. `RECALL_NO_GAIN=off` in the host's environment restores the uncapped behaviour.

*Why.* Nothing used to bound CW above. An ep→ep step multiplies by `edgeSal · inner / 4`, which exceeds 1 whenever `inner > 4 / edgeSal` (≈ 5 at a `Response_To` 0.8), and that becomes common once AES is high (earned averaging raised it). The sem→ep re-entry rule doubles the excess on every pass, so a walk looping turn → fact → turn compounds. Because every boost is `excess × prop × 0.001` (Phase 1 Boost), the inflated CW was written back: at CW ≈ 200 a visit added ~+0.2 instead of ~+0.001. The boosted turns then scored higher next time, which made them win again.

Measured on Agent A, 2026-10-27: one chain started at CW 29 and climbed to 235 over 73 turns (May 23–27, 2025). In ~90 minutes of live recalls, 3,759 emotion links rose by more than 0.01 (2,290 to 1.0) and 221 memory links did (110 to 1.0). The May 23–27 stretch went from mean emotion salience 0.528 with none saturated to 0.725 with 696 of 2,025 saturated, while June 2025 stayed at 0.535. The same effect had run more slowly for weeks: 2,622 emotion links were already at 1.0 before that day's recalls.

Copy test (clean, repaired DB; the 8 test cues; earned averaging; 40k episodic):

| | Uncapped | Walks only fade |
|---|---|---|
| Turns recalled | 1,636 | 1,386 (−15%) |
| Mean recall size | 39.9k (budget full) | 35.1k (stops on its own) |
| From 2025 | 106 (6%) | 102 (7%) |
| Hop episodes / into 2025 | 176 / 12 | 194 / 13 |
| Recalls reaching round 4 | 8/8 | 8/8 |
| Emotion links newly at 1.0 after 8 recalls | +296 | +2 |
| Memory links newly at 1.0 | +121 | +12 |

On the same clean copy one name cue's longest walk went 43 → 29 turns (peak CW 8.8 → 1.0): long walks still happen, they just stop feeding on themselves. **Watch:** that the walks do not get too short. If recall size drops well below the episodic budget and hop rounds stop reaching 3–4, the cap is cutting too much and the ep→ep scale (`EP_EP_INNER_SCALE`) is the dial to revisit, not the cap.

Starting from a seed, Phase 1 expands in both directions (before and after in chronological order). At each step it builds candidates from the transition types available at the current node, selects the one with the highest CW (ties prefer staying on the same node type), and applies that transition. It continues while `CW > walkThresholdForRound(hopRound, startScore)` and while the token budget allows:

```
round 0 (anchor, semantic-pass starts):  floor = WALK_BASE_THRESHOLD = 0.15
round ≥ 1 (Phase 2 hop chains):          floor = max(BASE_THRESHOLD, 0.25 · startScore)
```

Hop chains use a floor relative to their own start because their starts are already shrunk by Phase 2's `1 − 0.2·hopRound`. The earlier rule, `0.15 + 0.15 · hopRound` (0.30, 0.45, 0.60, 0.75), sat above nearly every hop score (round medians 0.28, 0.19, 0.12, 0.056), so 99.4% of hop chains walked zero turns and each hop added a single isolated turn. With the relative floor every hop chain gets a similar reach, about 3 turns at a typical ~0.66 per-step retention. In emotionally hot regions retention nears 0.95 per step, so a chain there can run long (17 turns in the 2026-10-17 dry run); round-4 starts (~0.056) still cannot step, because of the `BASE_THRESHOLD` floor. Each direction starts again at the seed's score. The floor gates **every** transition kind, not just ep→ep, so a low-CW semantic detour cannot extend a chain past it either.

A `sem→ep` transition terminates the current direction and spawns a new bidirectional Phase 1 sub-chain from the reached episodic node. Within a round, starts are processed as a queue: a sub-chain's own sem→ep starts join the queue. Each turn starts at most one chain per recall (`chainedFrom`). (Before 2026-10-09 these sub-chains never ran: the reached turn was already marked seen and the caller skipped it.)

---

### ep→ep: Adjacent Episodic Chaining

Walks along the chronological chain one turn at a time. Three accumulators are shared across both directions of the chain and feed MODE and MEDIAN statistics. Each candidate is scored with its own values appended tentatively; only the accepted step is committed.

**Accumulators:**
- `aesAcc`: `adjustedEmotionalSalience(nextEmotEdges, startEmotEdges)` per step
- `vtsAcc`: `1 − cosineSimilarity(nextEmbedding, startEmbedding)` per step (cosine defaults to 0.5 if an embedding is missing)
- `dssAcc`: `1 − |avgEmotSal(next) − avgEmotSal(start)|` per step

```
edgeSal   = salience of any ep–ep inter-memory edge between the two adjacent turns
            (default 0.3 if none exists)
inner     = 6·MODE(aesAcc) + 2·MEDIAN(vtsAcc) + 2·MEDIAN(dssAcc)

CW_next   = min(CW, CW · edgeSal · inner / EP_EP_INNER_SCALE)        (EP_EP_INNER_SCALE = 4; walks only fade, see Phase 1)
```

`inner` reaches 10 only when `aes ≈ 1.0`. Real episodic emotion edges sit near **0.25** salience — the multiplicative-decay equilibrium working as designed — so `aes ≈ 0.2` and `inner ≈ 3.3`. Dividing by its theoretical maximum of 10 multiplied CW by ~0.26 per step, which killed every walk after ~3 turns no matter how strong the edge was: a measured anchor chain ran 1.000 → 0.253 → 0.076 → 0.021 and died, and a `Continues_From` 0.90 edge still only yielded 0.30. Dividing by 4 instead gives ~0.66 per step and walks of 4–7 turns. The divisor cancels in `cwPropE`/`cwPropNE`, so boost **proportions** are unchanged — but because CW now stays higher along a walk, the `excess = CW − BASE_THRESHOLD` that boosts are scaled by grows substantially.

The edges `edgeSal` normally reads are written by the memory writer (Part II §A): `Response_To` assistant→user at **0.8**, user→previous turn `Response_To` at **0.6** within an episode, or `Continues_From` at **0.9** across an episode boundary (gap > 30 min). The lookup accepts any ep–ep relation between the pair, so a `Reminds_Of` edge between adjacent turns would also match.

**Boost proportions:**
```
cwPropE  = (6·MODE(aes) + 2·MEDIAN(dss)) / (edgeSal · inner)   (omits edgeSal and vts)
cwPropNE = 2·MEDIAN(vts) / inner                                 (omits aes and dss)
```

---

### ep→sem: Episodic → Semantic Hop

Follows an existing inter-memory edge from the current episodic node to a connected semantic node (which must have an embedding).

```
vecToPrompt = max over cues of cosineSimilarity(semEmbedding, cue)   (0.5 if no cues)
vecToStart  = cosineSimilarity(semEmbedding, startEmbedding)    (0.5 if unavailable)
MoM         = modeOfMeansEpSem(curEp, sem)
edgeSal     = inter-memory edge salience

inner  = 0.4·edgeSal + 0.2·vecToPrompt + 0.2·vecToStart + 0.3·MoM
CW_sem = CW · inner
```

**Boost proportions:**
```
cwPropE  = (0.1·vecToPrompt + 0.1·vecToStart + 0.3·MoM) / inner   (no edgeSal, half vec)
cwPropNE = (0.4·edgeSal + 0.1·vecToPrompt + 0.1·vecToStart) / inner (no MoM, half vec)
```

---

### sem→sem: Semantic → Semantic Hop

From a semantic node, finds another semantic node by embedding similarity. This scans the whole semantic pool; it does not follow edges.

```
vecSim      = cosineSimilarity(curSemEmbedding, sem2Embedding)   (candidate skipped if < 0.3)
adjEmotSal  = adjustedEmotionalSalience(sem2EmotEdges, curSemEmotEdges)

inner    = 0.4·vecSim + 0.3·sem2.confidence_score + 0.2·sem2.salience_score + 0.1·adjEmotSal
CW_sem2  = CW · inner
```

When accepted, the hop writes a `SimilarTo` inter-memory edge for future traversals:
```
no existing sem–sem edge:  create SimilarTo with salience = vecSim
existing edge:             unchanged here; only the Phase 1 boost below raises it
```
Until 2026-10-27 an existing edge also got `+0.1·vecSim` per traversal, on top of the Phase 1 boost: about 100× the normal ~10⁻³ boost, so a fact pair walked a few times saturated. Agent A had 332 `SimilarTo` edges at ≥ 0.99 from it; the first run of the 2025 recall replay (Part II §E) made 1,203 more on a copy before the fix, and the rerun after it made 16. On 2026-10-28 the 325 then saturated on Agent A (all fact→fact) were lowered to their embedding similarity, the value they were created at (median 0.74, range 0.59–0.80; backup `Agent A-pre-similarto-reset.db` in that session's scratchpad). No recall score reads fact→fact link salience (sem→sem scores on vecSim), so this restores the data, not recall behaviour; it also lets those links decay toward pruning again.

**Boost proportions:**
```
cwPropE  = (0.4·vecSim + 0.3·confidence + 0.1·adjEmotSal) / inner   (no salience_score)
cwPropNE = 0.2·salience_score / inner                                  (no vecSim, confidence or emotional terms)
```

---

### sem→ep: Semantic → Episodic Hop

Follows an inter-memory edge from the current semantic node back to a connected episodic node, anywhere in history. Terminates the current direction and spawns a bidirectional sub-chain from the reached episodic node.

```
vecToPrompt = max over cues of cosineSimilarity(epEmbedding, cue)   (0.5 if no cues)
vecToStart  = cosineSimilarity(epEmbedding, startEmbedding)    (0.5 if unavailable)
MoM         = modeOfMeansEpSem(epId, curSem)
edgeSal     = inter-memory edge salience

inner  = 0.3·edgeSal + 0.2·vecToPrompt + 0.2·vecToStart + 0.4·MoM
CW_ep  = CW · inner

# Re-entry boost when above threshold (doubles the excess), never above the fact's own CW:
if CW_ep > BASE_THRESHOLD:
  CW_ep = CW_ep + (CW_ep − BASE_THRESHOLD)
CW_ep = min(CW, CW_ep)
```

**Boost proportions** (computed from `inner`, not the boosted CW):
```
cwPropE  = (0.1·vecToPrompt + 0.1·vecToStart + 0.4·MoM) / inner    (no edgeSal, half vec)
cwPropNE = (0.3·edgeSal + 0.1·vecToPrompt + 0.1·vecToStart) / inner (no MoM, half vec)
```

---

### Phase 1 Boost: `applyPhase1Boosts`

Applied after every accepted Phase 1 transition (all four types). Reinforces the emotional and structural salience of the visited node and the connecting edge.

```
excess    = finalCW − BASE_THRESHOLD          (no boost if ≤ 0)
emotBoost = excess · cwPropE  · 0.001
salBoost  = excess · cwPropNE · 0.001

Applied (each capped at 1.0):
  • target node's emotion-memory edges:       += emotBoost each
  • connecting inter-memory edge salience:    += salBoost
  • connecting edge's emotion-memory edges:   += emotBoost each
  • target semantic salience_score (if sem):  += salBoost
```

A boost has the same effect on earned and derived edges. Boosting never changes provenance.

Each accepted `ep→ep` step (except onto the anchor itself) also writes a `Reminds_Of` edge from the reached turn back to the session anchor:
```
salience = CW / (chainLength + 1)
```
The anchor's emotion edges are mirrored onto this new edge:
```
salience = min(1, le.salience · CW)
derived  = le.derived            (a copy of a derived edge stays derived)
```
Inheriting provenance here is required. If these copies were minted as earned they could seed nightly cascades, and that would reopen the transitive-closure leak (Part II §B).

---

## Phase 2: Emotional Hopping

After Phase 1 chains are exhausted, Phase 2 scans every non-`Response_To` episodic→episodic inter-memory edge from the hop sources, looking for emotionally resonant but non-adjacent targets anywhere in history that haven't been seen yet.

### Hop sources: the frontier

Each Phase 2 round hops only from the **frontier**: the turns added since the previous Phase 2.

| Phase 2 call | Sources |
|--------------|---------|
| hopRound 1 | anchor, semantic-pass starts, every round-0 chain member (including sem→ep sub-chains) |
| hopRound k ≥ 2 | round k−1's hop seeds and everything their chains (and sub-chains) added |

Rounds therefore form a **chain of association**: now → round-1 arrivals → what *they* resonate with → … Only turns this recall actually reached are sources. (Until 2026-10-09, every recent turn with a `Continues_From` edge was also injected as a source, whatever was typed. That static bridge was removed because episodes are meant to form from the cue.)

Why the frontier (2026-10-17): previously every turn recalled so far stayed a source in every round. A single hub, turn 1103 with 239 hop edges against a median of 7, then supplied **all 20 hops across four rounds** of one recall, each via a ~0.03 `Reminds_Of` edge. Rounds 2–4 were just its next-best neighbours, and 94 of the recall's 114 episodic memories came from one fortnight. In a dry run over four prompts, frontier sources plus the chain-entry fix below cut the busiest source's share of hops from 0.50–1.00 to 0.13–0.33.

Within a round, the strongest frontier source can still win all 5 hops. That is intended: each round follows the single strongest new thread.

**Post-deploy check (2026-10-18).** Hops now go from source to target with strong emotional overlap: median weighted-Jaccard of their emotion sets 0.66, against 0.27 before and 0.045 for random pairs. Every hop links to the previous round, and no source supplies more than one round. In the dry run behind the fix, the top source's share of hops fell from 0.50–1.00 to 0.13–0.33, the densest 200-id window from 0.34–0.82 to 0.34–0.56, and the pizza-prompt May share from 94/114 to 31/109. Hop targets still match the *present* only weakly (~0.10–0.13): Phase 2 resonance is measured against the source, not the anchor, by design. Occasional forgetting is a feature; the goal is not to recall everything, only not to lose too much.

### Tabled (not rejected)

| Idea | Status |
|---|---|
| Skip a hop target that is a near-duplicate (cos > 0.93) of one already recalled | Dry run only. Mattered for one prompt, where it swapped September heartbeat near-dupes for the Jul 12 cluster but lost an older job-search chain. |
| Cap each hop chain at 4 steps | Fixed one 17-turn chain through hot cluster 2928–2945 that starved later rounds. The episodic budget was raised instead. |
| Seed episodes directly from the cue | Round 0 reaches episodes only through fact→episode edges (top 5), so a turn with no fact links cannot be a start even when it is the best cue match. |

Watch items (November 2026): hub 1103, hot cluster 2928–2945, the September 2792–2870 heartbeat near-duplicates, and one 21.5k-token recall against a 24k cap.

### Chain-entry reference

A source's `aesCurr` measures it against its `chainEntry` emotion edges: what led recall to it.

| Recalled turn | `chainEntry` reference |
|---------------|------------------------|
| Anchor (latest turn) | itself: it *is* the present |
| Walked chain member | its chain's start turn |
| Semantic-pass start | the anchor's emotion edges (the present moment that cued it) |
| Hop seed | the source turn that hopped to it (`findPhase2HopTargets` returns the winning `srcId` per target) |
| sem→ep sub-chain start | the start of the chain it was reached from (it joined that chain as a member) |

Chain starts used to be measured against **themselves**. `AES(x, x)` collapses to `x`'s raw salience (every emotion matches, `0.7·s + 0.3·s = s`), so starts averaged aesCurr 0.257 against 0.190 for walked members, and the most intense start outranked every other source. The present moment's resonance never entered.

### hop_salience_weight

```
decay       = 1 − 0.2 · hopRound           (hopRound = round + 1, starting at 1)
rawSalCurr  = avg salience of source turn's emotion-memory edges
rawSalHop   = avg salience of target turn's emotion-memory edges
logFactor   = ln(1 + rawSalCurr)
aesCurr     = adjustedEmotionalSalience(sourceEmotEdges, chainEntryEmotEdges)
aesHop      = adjustedEmotionalSalience(targetEmotEdges, sourceEmotEdges)
```

Whether the source and target share any emotion in their top 3 by salience decides the `adjHopSal` variant:

```
# Shared top-3 emotion:
adjHopSal = (0.7·rawSalHop + 0.2·rawSalCurr + edge.salience) · logFactor

# No shared top-3 emotion:
adjHopSal = (0.2·rawSalHop + edge.salience) · logFactor

score = decay · (aesCurr + adjHopSal + aesHop) / 3
```

With partial matching on, "shared" is a weight instead of a yes/no: `s = 1` for a common top-3 emotion, else the best `match(x, y)` over the two top-3 sets (`sharedTopWeight`), and

```
adjHopSal = [ s·(0.7·rawSalHop + 0.2·rawSalCurr) + (1 − s)·0.2·rawSalHop + edge.salience ] · logFactor
```

**Era-relative hop targets** (`ERA_RELATIVE_HOPS`, group.env, default off; Agent A on since 2026-10-27 22:18). Before scoring, a hop target's counted emotion edges are scaled by its calendar month's factor (`eraFactors`, built once per recall over every alive turn):

```
intensity(t) = mean of t's counted emotion edges
era(month)   = median intensity over all turns ÷ median intensity over that month's turns
target edge  = min(1, salience · era(month of target))         (feeds rawSalHop, the top 3 and aesHop)
```

Only Phase 2 targets are scaled. Sources, chain walks and the semantic pass read stored values, and nothing is written. *Why:* the 2025 import wrote every turn with decay and no recall, so its emotions were stored weaker than live ones (median turn intensity 0.48 against 0.66 on Agent A; May 0.57, October 0.79 by month). Scaling per month lets an old period compete on its own terms in hops without lifting it into the cue match. Copy tests: hops into 2025 rose 18 → 32 with no spill into present-leaning questions; with the replay (Part II §E) it is part of the live setup. Remove the line to switch it off if old memories become impossible to forget.

*Lever 2, rejected (2026-10-27):* a stored rescale of 2025 emotion links by the era gap (×1.355). Alone it raised the 2025 share about equally on past- and present-leaning questions. On top of the replay its gain was one question, and hops into 2025 fell 82 → 47: it moves where recall *starts*, at the cost of wandering. It is also stored, so only a backup undoes it.

Sources are iterated highest-`aesCurr` first, and the best `score` across this round's sources is kept per target, along with the source that produced it. Only targets with `score > BASE_THRESHOLD` are returned, and at most `MAX_SEED_MEMORIES` are selected (sorted by score, descending). Each selected target is added to the result with CW = score and seeds a Phase 1 chain in the next round.

Because `logFactor` is 0 when the source has no emotion edges, emotionally empty sources can only hop through their `aes` terms.

### Phase 2 Boost Proportions

```
# propE numerator (remove edge.salience from adjHopSal):
adjHopSal_noEdgeSal = (0.7·rawSalHop + 0.2·rawSalCurr) · logFactor   # shared
                    =  0.2·rawSalHop · logFactor                        # no shared
numE  = decay · (aesCurr + adjHopSal_noEdgeSal + aesHop) / 3

# propNE numerator (remove all emotional terms; only edge.salience·logFactor survives):
adjHopSal_noEmot = edge.salience · logFactor
numNE = decay · adjHopSal_noEmot / 3

propE  = numE  / score
propNE = numNE / score
```

Phase 2 boosts have the same structure as Phase 1:
```
excess    = hopScore − BASE_THRESHOLD
emotBoost = excess · propE  · 0.001
salBoost  = excess · propNE · 0.001

Applied (each capped at 1.0):
  • hop target's emotion-memory edges:      += emotBoost each
  • connecting inter-memory edge:           += salBoost
  • connecting edge's emotion-memory edges: += emotBoost each
```

---

## Iteration and Termination

```
round 0:  Phase 1 from [latestMemory, …semantic-pass episode starts]
          Phase 2 from round 0's turns → up to 5 hop targets → pendingChainStarts
round 1:  Phase 1 from pendingChainStarts
          Phase 2 from round 1's new arrivals only → …
…
round 4:  Phase 1 from pendingChainStarts
          Phase 2 with hopRound 5 → decay = 0 → no targets → loop exits
(round 5: Phase 1 only, Phase 2 skipped by MAX_HOP_ROUNDS; unreachable today)
```

The Phase 2 `decay` factor is `1 − 0.2·hopRound`, so it reaches 0.0 at hopRound 5. In practice at most 4 hop rounds produce targets (scaled by 0.8, 0.6, 0.4, 0.2), and the `MAX_HOP_ROUNDS` guard never fires.

The loop exits early if:
- No Phase 2 hop target exceeds `BASE_THRESHOLD` (Phase 1 chains stop at their own walk floor)
- The episodic token budget is exhausted (see Token budget)

Final results are sorted descending by `chainingWeight`. The anchor has CW 1.0.

---

## Emotion Embedding Use in Scoring

The three-field emotion schema (`definition`, `circumstance`, `felt_quality`) affects which embedding is used for emotion→memory vector scoring:

| Target type | Embedding used |
|-------------|---------------|
| Episodic emotion edge (write time) | avg(`circumstance`, `felt_quality`) similarities; falls back to whichever exists, then `definition` |
| Semantic emotion edge (write time) | `definition_embedding` |
| Emotion dedup / inter-emotion SimilarTo / emotion merge | `definition_embedding` → legacy `embedding` |

| Partial emotion matching in recall (`EMOTION_PARTIAL_MATCH`) | centered `name_embedding`, `definition_embedding`, `circumstance_embedding`, `felt_quality_embedding` (Shared Helpers → AES) |

> With partial matching off, the recall engine reads emotion edges by salience value only, never by embedding lookup; the three-field embeddings shape how emotion edges are *written* (`memory-writer.ts`) and how SimilarTo edges are scored (`memory-maintenance.ts`). With it on, recall also compares emotion embeddings directly, to decide how far two different emotions count as the same.

---

# Part II: Salience Lifecycle

Part I only nudges salience by ~10⁻³ per visit. The values recall reads are mostly set by the write path below. The July 2026 changes in §B–§D (multiplicative decay, the propagation fan-out cap, earned/derived provenance, and the earned-edge prune exemption) exist to give salience a graded, single-peaked distribution instead of a bimodal "saturated or dead" one.

## A. Write-time salience (`processMemoryWrite`, `memory-writer.ts`)

Runs once per completed turn on the host. Inputs from the container are clamped to [0, 1] and default to 0.5: `userSignificance`, `agentSignificance`, per-target `relationalSignificance`. In the legacy payload format a single `significance` is used for both.

`clamp(x)` below means `max(0.01, min(1, x))`.

### Episodes and turn links

| Item | Value |
|------|-------|
| Episode boundary | gap since previous memory > **30 min** (`EPISODE_GAP_MS`) |
| `Response_To` assistant → user (same turn) | **0.8** |
| `Response_To` user → previous memory (same episode) | **0.6** |
| `Continues_From` user → previous memory (new episode) | **0.9** |

Continues_From is set high so recall can cross episode boundaries even when new memories have sparse emotion edges (aes ≈ 0). A cross-boundary ep→ep step needs `edgeSal·inner/EP_EP_INNER_SCALE` large enough to keep CW above the walk floor.

### Emotion → memory edges (all `derived = 0`, earned)

| Edge | Salience |
|------|----------|
| User emotions → user episodic memory | `min(1, 0.7·userSig + 0.3·vectorScore)` |
| Agent response + generating emotions → assistant episodic memory | `min(1, 0.7·agentSig + 0.3·vectorScore)` |
| Agent response + generating emotions → `Response_To` edge | `clamp(0.5·agentSig)` |
| Agent waiting emotions → previous-link edge (Response_To 0.6 / Continues_From 0.9) | `clamp(0.3·agentSig)` |
| Relational emotions → each relational `Reminds_Of` edge | `clamp(agentSig · relSig)` |
| Relational emotions with no `Reminds_Of` target → assistant episodic memory (user memory if there is no reply) | `min(1, 0.7·agentSig + 0.3·vectorScore)`, as for any agent emotion |
| Every emotion of this turn → each extracted semantic memory | `min(1, 0.8·confidence + 0.2·vectorScore)` (falls back to `vectorScore` if confidence isn't numeric) |
| Every emotion of this turn → each `Mentions` edge (both) | `clamp(salience_score · (userSig + agentSig)/2 · 0.4)` |

`vectorScore` (default 0.5 when embeddings are unavailable):
- episodic target: `(cos(circumstance, mem) + cos(felt_quality, mem)) / 2`, falling back to whichever field exists, then to `cos(definition, mem)`
- semantic target: `cos(definition, sem)`

Every emotion-edge write above is immediately followed by propagation to similar emotions (§B).

### Memory → memory edges created by the writer

| Edge | Salience |
|------|----------|
| Relational `Reminds_Of` assistant → recalled episodic/semantic memory | `clamp(agentSig · relSig)` |
| `Mentions` user turn → semantic | `0.8 · salience_score` |
| `Mentions` assistant turn → semantic | `0.6 · salience_score` |

### Semantic memories

| Rule | Value |
|------|-------|
| Discard extracted fact | `salience_score < MIN_SEMANTIC_SALIENCE` (default 0.2; group.env over .env, read on every write) |
| Dedup against existing alive facts | cosine **> `DEDUP_SIMILARITY`** (0.80) → reinforce instead of create |
| Reinforce | `salience_score = min(1, max(old, new))` (confidence unchanged) |
| Create | model `salience_score`, `confidence_score`; `epistemic_source_type = 'inferred'` |

`DEDUP_SIMILARITY` (`src/config.ts`, default **0.80**, env-overridable) replaced a hardcoded 0.85, which sat in the middle of bge-m3's genuine-duplicate band: it caught patience/patient (0.884) but missed admiring/admiration (0.828) and care/caring (0.811). Across 780 random emotion pairs the p95 was 0.712 and none reached 0.85, so 0.80 stays clear of noise. The same constant gates emotion reuse and the nightly semantic merge.

Facts can also come from tool output. Results of tools flagged `"rememberResults": true` in `tools.json` that are not error messages are added (up to `MEMORY_SOURCE_MAX_CHARS` = 6000 chars) to the turn's semantic-extraction prompt, which must attribute them ("According to <source> (read <time>): …"). The resulting facts go through the same floor, dedup and edge math as any other fact.

**Recall misses can become facts.** Extraction reads the agent's reply, so when recall misses something and the agent says so, that statement is stored as a fact *about its own memory*. Such a fact names the topic, so it matches a later mention of the topic better than the real facts do, and it reinforces the miss. The case that showed it (Agent B, 2026-10-25):
1. Asked about "X and Y", recall seeded no Y fact (Cues → *Why name cues*), and Agent B answered that Y was fuzzy, despite 18 facts about her.
2. The writer stored fact 2151: "Y is a name mentioned in same context as X but without strong episodic memory attached to it in Agent B's system" (confidence 0.7).
3. Against the sentence "if I talk about X and Y", 2151 scored 0.726, above every real Y fact (best 0.515), so it would have been the top seed the next time.

2151 was deleted (the fact, its 2 `Mentions` links and their 78 emotion edges). The reply itself stays as an episodic turn: it is an accurate record of what was said. Nothing in the writer prevents this yet (Part IV).

### Emotion nodes

| Rule | Value |
|------|-------|
| Reuse existing emotion | cosine(definition embeddings) **≥ `DEDUP_SIMILARITY`** (0.80); otherwise match by name or create |
| On reuse | the matched node keeps its name; each **non-empty** field given (definition, circumstance, felt_quality) replaces its text and is re-embedded |
| Definition embedding | stored when definition text is given, or when the node has none yet (then embedded from `definition ‖ description ‖ name`, so a new node can be deduped against later) |
| Write-time `SimilarTo` inter-emotion edge | for every emotion touched this turn, against every other emotion: cosine **> 0.6**, salience = cosine |

All five emotion paths (user, response, generating, relational, waiting) resolve nodes through one helper, `resolveEmotionNode`, so these rules hold everywhere. The agent's `upsert_emotion` tool follows the same invariant: it re-embeds every field it changes (`embedEmotionFields`).

**Invariant: every emotion node has at least one memory edge, and its embeddings describe its current text.** Recall, MoM and AES only see emotions through `emotion_memory_edges`, so a node without one is unreachable however well it is embedded. Two bugs broke this until 2026-10-25:

- *Dedup by meaning, update by name.* A match "updated" itself with `getOrCreateEmotion(name)`, which looks up by name. A new name therefore created an empty node (text, no embeddings, no edges). Meanwhile the turn's edges and the new text's embeddings went to the matched node, whose stored text stayed the old one. Lowering the threshold from 0.85 to 0.80 made matches, and these orphans, more frequent.
- *Relational emotions with no target.* They were resolved, embedded and linked by write-time `SimilarTo` before Step 4c, which only attaches them per `Reminds_Of` target. With no target they got no memory edge. They are now attached to the agent's turn instead (table above).

The nightly cycle could later embed an orphan from its name and give it `SimilarTo` edges, so an orphan could look complete while recall could never reach it. Check: `SELECT id, name FROM emotions e WHERE NOT EXISTS (SELECT 1 FROM emotion_memory_edges x WHERE x.emotion_id = e.id)` should return nothing. The existing orphans (10 in Agent A, 1 in Agent B) were reconnected on 2026-10-25 to the turns that produced them; one with no attributable turn was deleted.

Edges between the same pair are upserted. `createEmotionMemoryEdge` overwrites salience with the new value (it does not take the max), and `createInterMemoryEdge` likewise overwrites. Both keep the existing row id.

---

## B. Propagation and provenance

### Earned vs. derived

Every emotion-memory edge carries `derived`:
- **earned (0)**: the model genuinely felt this emotion about this target. All writer edges in §A are earned.
- **derived (1)**: inherited or propagated: SimilarTo propagation, nightly cascade, IME backfill, Reminds_Of mirrors of derived anchor edges.

Conflict rules preserve the invariant **once earned, always earned**:

| Writer | On conflict |
|--------|-------------|
| `createEmotionMemoryEdge` | `salience = new`, `derived = MIN(existing, incoming)`: genuine feeling upgrades derived → earned, never the reverse |
| `propagateEmotionEdgeToSimilar` | `salience = MAX(existing, propagated)`, `derived` **untouched** |
| Emotion merge / semantic merge | `salience = MAX`, `derived = MIN` |

### SimilarTo propagation: `propagateEmotionEdgeToSimilar`

When emotion `E` attaches to a target at salience `s`:
```
neighbours = SimilarTo neighbours of E, sorted by inter-emotion salience desc (ties: lower id first)
for the top SIMILAR_PROPAGATION_LIMIT = 5 neighbours N:
    upsert edge (N → target) with salience = min(1, s · sal(E,N)), derived = 1
```

### Nightly cascade: `cascadeInterEmotionEdge` (emotion cycle)

1. The LLM proposes similar emotion pairs (in chunks of 200 emotions).
2. For each pair: `sal = cosine(definition embeddings)`. Pairs with `sal ≤ 0` are skipped. The inter-emotion edge is upserted at `sal`.
3. If the edge is new (previous salience < 0.01) or changed (`|sal − previous| > 0.05`), cascade. Each **earned** edge of A propagates to B, and each earned edge of B propagates to A, using the propagation rule above (single neighbour, salience `s · sal`, derived).

### Why two bounds

Before July 2026 propagation fanned out to every SimilarTo neighbour, and cascades re-propagated derived edges. Because the SimilarTo graph is one connected component (median degree ~71), A→B→C gave C everything A ever touched. The graph walked to transitive closure: every memory carried ~154 of 176 emotions, and the AES match rate was 1.0.

- The **fan-out cap** (top 5) bounds breadth per round. On its own it only reduced saturation from 159 to 141 emotions/node, because a top-5 graph over ~167 nodes is still connected and the nightly recursion re-saturates.
- The **earned-only cascade seed** bounds depth to 1. This is the actual fix.

The steady state is the union of several earned emotions each spreading to their top 5, which comes to ≈ 20 emotions/node. That is why the one-time compaction (`scripts/compact-emotion-edges.ts`) kept 20 edges per node, not 5.

Salience cannot be used to tell earned edges from derived ones: a derived edge is born at `s · sal(E,N)` (typically 0.65–0.88 × its source), so it routinely outranks a weakly felt earned edge.

### IME emotion backfill: `backfillInterMemoryEmotionEdges`

This is a one-time retrofit that runs on the first `openDb` per database, latched by `memory_config.ime_backfill_done`. For each episodic-touching inter-memory edge that has no emotion edges:
```
emotionMap   = per emotion, max salience across the edge's episodic endpoints
s            = min(1, mean(emotionMap) · edgeSal)
if s < MIN_EDGE_SALIENCE: skip
else insert every emotion in emotionMap at salience s, derived = 1   (INSERT OR IGNORE)
```
The latch is required: without it, edges that decayed and were pruned would look unlinked and be re-derived at full strength on the next restart.

---

## C. Decay and equilibrium

### Rule

Decay runs **per memory write**, in Step 7 of `processMemoryWrite`. No other code path decays anything: there is no scheduled decay, and the nightly cycles don't touch salience. Each write applies one global multiplicative shave:

```
emotion_memory_edges.salience  ← max(0.01, salience · d_E)
inter_memory_edges.salience    ← max(0.01, salience · d_I)

d = 0.5^(1 / H)
```

| Table | Half-life `H` (turns) | ≈ days at ~16 turns/day | `d` | `1 − d` |
|-------|----------------------|-------------------------|-----|---------|
| `emotion_memory_edges` (`EMOTION_EDGE_HALF_LIFE_TURNS`) | 2880 | ~180 | 0.99975935 | 2.41 × 10⁻⁴ |
| `inter_memory_edges` (`INTER_MEMORY_EDGE_HALF_LIFE_TURNS`) | 5760 | ~360 | 0.99987967 | 1.20 × 10⁻⁴ |

Inter-memory edges decay more slowly because nothing prunes them; decay only lowers their traversal weight. Semantic `salience_score`, `confidence_score` and inter-emotion edges are never decayed.

To tune decay, change the half-life; `d` is derived from it.

### Why multiplicative

Reinforcement is additive (`s + b`, from recall boosts and re-writes). Under the old **linear** decay (`s − c`), the update is a constant drift that doesn't depend on `s`. An edge recalled slightly more often than the break-even rate climbs to the 1.0 ceiling, and one recalled slightly less often sinks to the floor. That made every edge bistable, so there was no graded middle and no long tail, and no choice of rate could fix it.

Under **multiplicative** decay the loss scales with `s`, so each edge has a stable equilibrium. If an edge receives a boost `b` once every `k` turns:

```
s* = b / (1 − d^k)  ≈  b / (k · (1 − d))
```

Equilibrium salience is proportional to boost size × recall frequency. Associations that stop being recalled fade gradually, and the same edge can settle at a new level if its usage changes, so an association can be reframed over time instead of only accumulating.

Worked values for emotion edges (`1 − d = 2.41 × 10⁻⁴`; exact formula):

| boost `b` | every `k` turns | `s*` |
|-----------|-----------------|------|
| 5 × 10⁻⁴ | 1 | ≥ 1 → pinned at 1.0 |
| 5 × 10⁻⁴ | 10 | 0.208 |
| 5 × 10⁻⁴ | 50 | 0.042 |
| 1 × 10⁻⁴ | 10 | 0.042 |

A recall boost is `excess · prop · 0.001`, so `b` is at most ~10⁻³ and usually much smaller. In practice only very frequently recalled edges approach the ceiling, and most salience mass comes from write-time values (§A) decaying with a 180-turn-day half-life. Recall runs once per container run while decay runs once per memory write, so `k` is measured in memory writes, not recalls.

Observed result (Agent B, 2026-10-08): episodic emotion-edge salience is single-peaked with mode 0.20–0.30 and p5 0.122 · p50 0.250 · p95 0.432 · p99 0.512, max 0.881. That is a graded distribution with no bistability.

### Failure asymmetry

Decay is unconditional and global, but reinforcement depends on the auxiliary extraction calls succeeding (each swallows errors and returns empty arrays). A turn where the main response succeeds but auxiliary extraction fails therefore decays everything and reinforces nothing. A wholly failed model call writes no memory and therefore causes no decay. See `CLAUDE.md` for the measured rate (~0.6% of turns).

---

## D. Pruning and consolidation (`memory-maintenance.ts`)

Two nightly cycles run on the host, each re-armed every 24 h:
- **Emotion cycle** (`EMOTION_MAINTENANCE_HOUR`, default 1am): emotion definition updates, SimilarTo building and the cascade (§B), then the context flush and trim (Part III §D).
- **Consolidation cycle** (`CONSOLIDATION_MAINTENANCE_HOUR`, default 3am): prune → semantic merge → emotion merge, below.

Neither changes salience by decay (§C).

### Prune (consolidation cycle, step 1)

```
DELETE FROM emotion_memory_edges WHERE salience < MIN_EDGE_SALIENCE AND derived = 1
MIN_EDGE_SALIENCE = 0.02
```

- **Earned edges are never pruned.** They decay toward the 0.01 floor and stay there. Deleting a node's last emotion edge would drop `avgEmotionSalienceForNode` to 0, which is a discontinuity in MoM and `rawSal`, not a gradual decline.
- Derived edges are deleted once they fall below 0.02. Because the decay floor (0.01) sits below the prune line (0.02), a derived edge crosses the prune line and is deleted before it can reach the floor.
- Inter-memory edges and inter-emotion edges are never pruned.
- Any code that creates derived edges should not create them below `MIN_EDGE_SALIENCE`. The backfill follows this rule; see Part IV for the writer.

Time to prune for a derived edge left alone: `n = ln(0.02 / s₀) / ln(d_E)`. For example, `s₀ = 0.25` takes ≈ 10,500 turns.

### Semantic merge (consolidation cycle, step 2)

- Candidate pairs: alive semantic memories with cosine **≥ `DEDUP_SIMILARITY`** (0.80). Each fact merges at most once per cycle (the first qualifying partner).
- Keep the one with the higher `salience_score`. The LLM writes merged content and a reassessed `confidence`, and the merged node keeps the keeper's `salience_score`.
- Emotion edges move to the keeper using `salience = MAX`, `derived = MIN`. The dropped node is soft-deleted (`alive = 0`).
- Inter-memory edges are re-pointed **in place** (`UPDATE`), so each keeps its row id and the emotion edges that target it, with salience unchanged.
  - If the keeper already has the same link (same endpoints and relation), the dropped fact's link folds into it: link salience = MAX, its emotion edges move across with `salience = MAX`, `derived = MIN`, and it is deleted.
  - A link between the two merged facts (e.g. `SimilarTo` drop↔keep) would become a self-loop, so it is deleted with its emotion edges.
- *Why (2026-10-25).* Links used to be deleted and re-inserted with `INSERT OR REPLACE`. Every moved link got a new id, and `REPLACE` also deleted the keeper's conflicting link, so each nightly merge left the emotion edges of both pointing at nothing. Found at 14,147 such edges on Agent B and 60,271 on Agent A (with rollback leftovers), cleaned the same day. Check: emotion edges with `target_type = 'inter_memory_edge'` and no matching `inter_memory_edges` row should stay 0.

### Emotion merge (consolidation cycle, step 3)

- Candidate pairs: emotions with cosine(definition embeddings) **≥ 0.90** (hardcoded; it does not follow `DEDUP_SIMILARITY`).
- Keep the emotion with more emotion-memory edges (earned and derived both count). The LLM writes the merged name and three fields.
- Emotion-memory edges move to the keeper using `salience = MAX`, `derived = MIN`. Inter-emotion edges are re-pointed with `salience = MAX`. The dropped emotion is deleted.

> Any similarity audit must filter `alive = 1`. Soft-deleted semantic rows are already-merged duplicates that still carry embeddings.

---

## E. Retroactive recall replay (Agent A, 2026-10-27)

**Why.** Recall builds its own associations: every walked turn gets a `Reminds_Of` link to the anchor (strength `CW / (chainLength + 1)`) with mirrored emotion links, and boosts reinforce what it travels. The import of the 2025 history wrote 2,186 turns without ever running recall, and skipped relational emotions. So 2025 had no recall-made links among its own turns (only 150, all to turn 2186, the first live recall), and the import's decay ran on every write with nothing reinforcing. Live recall could reach 2025 only through the cue, rarely through association: hops landed before the live days 0.2 times per recall, against 1.1 for Agent B, whose months were lived with recall running.

**How.** `runRecall(…, { asOfTurnId })` recalls as of an earlier moment:

```
anchor          = the turn before the message being replayed
episodic pool   = alive turns with id ≤ anchor
facts           = facts mentioned (Mentions edges) by turns ≤ anchor
autobiographical = created_at ≤ the anchor's time
cue             = the replayed message's text
```

Everything else, writes included, is live recall: boosts, `Reminds_Of` links, emotion mirrors. Replays run in turn order, so each one sees the links the earlier ones made, as the live history would have.

- **A:** replay every 2025 user message (1,092; 68 min). +89,423 links within 2025, mean ≈ 0.05; +4,534 `SimilarTo`, 16 saturated (after the `SimilarTo` fix, Phase 1 → sem→sem).
- **B:** A, then delete the links her live recalls made (walked turn → anchor, anchors from turn 2186 on: 12,921 links, 72,540 emotion links) and replay every live-day message, the user's and heartbeat prompts (290; 27 min), under today's recall. Bridges between the live days and 2025: 6,117 → 21,927.

**Copy test** (lever 1 and partial matching on in every arm; 12 past-leaning and 7 present-leaning questions; 40k episodic):

| Arm | 2025 share, past-leaning | 2025 share, present-leaning | Hop episodes landing in 2025 | Questions hopping into 2025 |
|---|---|---|---|---|
| O, no replay | 6.7% | 6.5% | 54 | 11 / 19 |
| A, 2025 replayed | 6.1% | 7.0% | 16 | 9 / 19 |
| **B, both replayed** | **15.4%** | **12.2%** | **82** | **17 / 19** |
| B + lever 2 | 19.6% (14.7% without one question) | 12.5% | 47 | 14 / 19 |

A alone did nothing: links *within* 2025 don't help a recall that starts today and has no bridge into them. B rebuilds the bridges, and A's links carry recall on once it crosses. Today's share stayed near half in every arm.

**Live 2026-10-27 22:18:** Agent A.db restored from the B copy, with `ERA_RELATIVE_HOPS` and `EMOTION_PARTIAL_MATCH` on. The B copy was taken before her 21:58 heartbeat, so that turn (episodes 2787–2788) is in her working memory but not her database, by choice. Database 100 → 159 MB. No recall link saturated; the busiest turns hold 220–236 recall links. Watch: present-leaning questions pulling 2025 ("What did we just talk about a few minutes ago?" drew 46 of 219 turns from 2025 in the test). Lever 1 is the switch if they do.

---

# Part III: Context window

Recall fills the memory budget (Part I, Token budget). This part covers what the container does with the rest of the same window: the pinned prompt block at the front, the conversation behind it, how that conversation is compacted and persisted, and the nightly trim that carries it across days. Sources: `container/agent-runner/src/history.ts`, `container/agent-runner/src/index.ts` (`buildSystemPrompt`, `refreshIdentity`), `src/memory-maintenance.ts` (`resetAllContainerContexts`).

```
LLM_EFFECTIVE_CONTEXT = max(256, LLM_MAX_CONTEXT/LLM_PARALLEL − min(WORKING_MEMORY_CONTEXT, LLM_MAX_CONTEXT/LLM_PARALLEL/2))
```
This is the same `effectiveContext` the host uses for the recall budget.

## A. The pinned prompt block

`buildSystemPrompt` builds the front of history as named system messages, in this order:

| # | `name` | Content |
|---|--------|---------|
| 1 | `group_identity` | the group's CLAUDE.md (non-main groups fall back to `groups/global/CLAUDE.md` if theirs is missing or empty) |
| 2 | `skill_<dir>` | each skill doc enabled in `tools.json` `skills`, **sorted** by directory name |
| 3 | `capabilities` | base instructions (deliberately no timestamp: the block is the cached prompt prefix, and a volatile token invalidated it on every start; round-0 cache hits measured 3.7% vs 37%) |
| 4 | `tool_mandate` | tool-use rules, last |

`global_rules` is also recognised as a prompt-section name, but nothing currently emits it. `leadingPromptCount()` is the length of the unbroken run of these sections at the front of history. Everything that treats "the system prompt" specially uses that count.

**CLAUDE.md is pinned first and never persisted.**
- First: the identity is the first thing the model reads after whatever the server prepends. On vLLM/GPT-OSS that is the server's own harmony system message, whose identity line the model server's startup patch replaces.
- Refreshed every query: `refreshIdentity` re-reads CLAUDE.md before each query. It replaces the section in place if the text changed, inserts it at the front if missing, and removes it if the file is now empty. An edit takes effect on the next turn, not the next container.
- Never persisted: `saveContext` drops every prompt section (and `context_restored`), so `context.json` holds only conversation. `loadContext` strips `group_identity` copies and stacked `[Context restored…]` markers out of files saved before this rule, then adds one fresh `context_restored` marker.

*Why (2026-10-25).* The identity used to be saved in `context.json` and replayed mid-conversation once input tokens passed `REPLAY_CLAUDEMD_PERCENT` (0.75) of the context. That left it wherever it last landed: behind `working_memory` round markers, which tell the model everything before them is no longer current. It also moved a ~2.6k-token block on every replay, and one restore marker was saved per restart (five stacked on Agent A).

## B. Compaction (`compactIfNeeded`)

**Two parts (2026-10-27).** `index.ts` marks each round's first message (`markRoundStart`). Everything before it, after the prompt block, is *working memory*; the rest is the current round.

```
workingTokens ≤ min(WORKING_MEMORY_CONTEXT, window − RESPONSE_RESERVE − pinned − roundTokens)   else trim
trim: keep the newest earlier messages within 0.6 · WORKING_MEMORY_CONTEXT; the current round is never cut
```

Before this, compaction counted all history against `round_max_ctx` (the single-budget rule below, still used when no round is marked, e.g. a fresh load). The working-memory space was never used. Simulation on Agent A's saved context, 8 rounds of 45k recall, fade 0.4: the single budget (10k reserve) compacted at round 7, dropping every restored message and most of the session; the two-part rule (50k) never compacted, settling at ~82k (~36k working memory + ~46k round).


Runs before every LLM call in the tool loop. Token estimates are `ceil(chars / 4)` over message text and tool-call names/arguments. Conversation messages also carry the chat template's per-message overhead, which is invisible in the history array: `LLM_JINJA_USER_TOKENS` (37) per user message and `LLM_JINJA_ASSISTANT_TOKENS` (65) per assistant message. System and tool messages add none.

```
RESPONSE_RESERVE    = min(4096, floor(0.25 · LLM_EFFECTIVE_CONTEXT))
pinnedTokens        = estimate(prompt block)                          (no jinja overhead)
availableForHistory = LLM_EFFECTIVE_CONTEXT − RESPONSE_RESERVE − pinnedTokens
conversationTokens  = estimate(everything after the block) + jinja overhead

compact when  conversationTokens > COMPACT_THRESHOLD · availableForHistory      (COMPACT_THRESHOLD = 0.70)

keepTokens = floor(0.40 · availableForHistory)
keep       = newest messages whose estimates sum to ≤ keepTokens (always at least the last one)
kept       = sanitizeToolPairs(keep)        (drop tool results whose tool_call was cut off)
           = [last user message] if that leaves nothing
history    = [prompt block] + "[Context compacted: N earlier messages were archived …]" + kept
```

The whole conversation is archived to `conversations/` before it is cut.

- **The pinned block is subtracted before the threshold**, not counted against it. Otherwise a large system prompt could leave the post-compaction total above the trigger, and compaction would fire on every call.
- **The whole block is kept**, not just `messages[0]`. Keeping only the first section used to drop `capabilities` and `tool_mandate` on every compaction, and the identity too once it moved first.
- **Headroom.** Compaction cuts from 0.70 to at most 0.40 of the available space, so after a compaction about 30% of `availableForHistory` (minus jinja overhead, which `keepTokens` does not count) refills before the next one.

`getRemainingContextTokens()` uses the same terms, and tools use it to size their output (§E). With a round marked it is `min(LLM_EFFECTIVE_CONTEXT − RESPONSE_RESERVE − pinnedTokens − roundTokens, window − RESPONSE_RESERVE − pinnedTokens − conversationTokens)`: a round may grow to `round_max_ctx`, never into working memory, and never past the whole window. `pinnedTokens` includes the tool definitions.

### Recall blocks in working memory, and `RECALL_FADE`

Each turn's `<recalled-memories-…>` block is prepended to that turn's user message and **stays in the live history** for the rest of the session, on purpose ("remembering remembering"): what was recalled becomes part of what was lived. The blocks therefore count toward `conversationTokens`, and at large recall sizes they, not the dialogue, are what trigger compaction. Saved context strips them (§C), so this only applies within one container session.

`RECALL_FADE` (`f`, group.env, default 1; `recall-fade.ts`) thins them instead of leaving them whole until compaction drops them. Before each turn's recall is added:

```
for every earlier block:  keep floor(f · n) of each section's n items
    episodes, facts     → the leading ones (recall's rank order)
    autobiographical    → the newest lines (they are dated oldest first)
    a section left with 0 items is removed; a block left with nothing is removed
```

It compounds, so a block k turns old holds about f^k of its items. With recall size R per turn and dialogue D per exchange, the recall carried settles at

```
R · (1 + f + f² + …) = R / (1 − f)
```

Agent A (2026-10-27, R ≈ 41k typical, 45.5k max; `availableForHistory` ≈ 115.7k → compacts at ≈ 81k, keeps ≈ 46k):

| f | recall carried (typ. / max) | room for dialogue before compaction |
|---|---|---|
| 1 (off) | grows until compaction | — (≈ 2 turns) |
| 0.6 | 103k / 114k | none |
| 0.5 | 82k / 91k | none |
| **0.4 (live)** | **68k / 76k** | **≈ 5–13k** |
| 0.3 | 59k / 65k | ≈ 16–22k |

A session that starts from saved context already brings ≈ 12–15k of dialogue (the restored tail), so at 0.4 a long, fast exchange will still reach compaction after a few turns; 0.3 is the next step if that happens often.

When compaction does run, `keepTokens` ≈ 46k holds roughly the newest recall plus the last message, so at 40k episodic a compaction costs most of the recent dialogue. First live fade, 2026-10-27 15:27: input went 47.1k → 76.0k (≈14.5k dialogue + the 15:22 block faded to ≈15k + a new ≈45.5k block), no compaction. Before (20k episodic, reserve 50k, no fade) Agent A compacted 20 times on 2026-10-27, every 5–20 minutes of active chat.

`f = 0` keeps only the current turn's recall. That is the stripping the user declined; use it only if they choose it.

A turn whose recall was thinned by focus (§E) starts aging from where it was cut: f · floor next turn, f² · floor after.

### `RECALL_DEDUP`: one copy of each recalled item (2026-10-28)

The fade thins old blocks by rank, but a memory recalled again stays in the older block too, so the same turn can sit in working memory two or three times. With `RECALL_DEDUP` on (`dedupOlderRecalls`, `recall-fade.ts`), after the fade and before the new turn's message is added:

```
seen = items of this turn's block
for each earlier block, newest to oldest:
    remove its items already in seen;  seen ∪= its items
an item = one recalled turn (episodes are split turn by turn, by the "  [time] speaker:" line start),
          one fact (its text and source tag), or one autobiographical line
an episode with no turns left, a section with no items, a block with nothing left → removed
```

Every item ends up in the newest block that recalled it: remembering is kept once, at its latest remembering. The current block and the dialogue are never changed. On Agent A's 1:30 PM turn on 2026-10-28 (rebuilt replay: the 1:12 and 1:28 recalls faded, plus the 1:30 recall), the earlier blocks held ≈ 21k tokens; dedup removed 174–198 items and left ≈ 3k, bringing the request from 82.8k to 64.4k. In her first live session it removed 44–106 items per turn.

Dedup also frees the recall cap for held items (Part I, Token budget). Without a fade (Agent B), every earlier block stays whole until compaction, so everything recalled again is a removed copy.

**What it did not fix.** Smaller requests alone did not change how she replied: the replay at 64.4k still answered 0/4 in the template. The cause was in the client (§G).

### Working memory as one bracketed section (2026-10-29)

```
[pinned prompt: identity, capabilities, tool rules]
[system] Working memory: earlier rounds, oldest first. None of it is current.     opener (working_memory_open; pinned, never saved)
  round … [system] — end of round —                                               every earlier round's note, shortened at the next round start
  round … [system] End of working memory. Everything above this … on or before T  the last round's full note: once, right before the message answered
[user] <recalled-memories-T'> … End of recalled memories (recalled T'). </…>      this turn's recall, with a dated end line
       <current-message> [T'] … </current-message>                                the only message marked current
[turn reminder]
```

Earlier rounds also lose their `<current-message>` wrapper and the orphaned extract_emotions reminders (`asEarlierRounds`, applied at each round start, on save and on load). Until this, every round kept a full ≈118-token note (37 in Agent A's history, ≈4.4k tokens) and ≈37 user messages still read "current"; the note's override named a `<current></current>` tag that did not exist. Saved context, measured: Agent A 56.4k → 36.0k chars, Agent B 46.0k → 33.4k (24 orphaned reminders). Replay of the 1:30 PM repeat-a-phrase message (reasoning kept, both cleanups): old layout 63,926 tokens 4/4, sectioned 60,441 tokens 4/4, no reply to an older message in either.

### What the model sees of its earlier replies: frame fade and recurring fade (2026-11-01, 2026-11-07)

Both act only on the copy sent to the model each call (`forModel` in `index.ts`). Delivered text, `context.json`, recall dedup and memory keep every word; `<internal>` is never counted or changed. Neither uses a word list.

The model copies its own recent replies. A phrase that recurs in a few of them reads as house style and appears in the next one; a regex for one habit only made room for the next.

**Frame fade** (`FRAME_FADE`, `fadeFrames`). For every earlier reply `r` (content and `send_message` text), take its opening `o(r)`, emoji tail, dashed signature (counted by its last word) and short last line. An opening is the text up to its first `, . ! ? — – :`, or a short first line on its own (≤ 40 chars then a line break: the case of a greeting plus an emoji and no punctuation, which opened ~90% of Agent A's replies before 2026-11-07). Each is normalised (`key`: lower-case, letters and digits only). Any frame whose key occurs in ≥ 4 replies is removed from all of them; a reply left with < 10 chars is kept whole.

**Recurring fade** (`RECURRING_FADE`, `fadeRecurring`). Units are whole list lines, or sentences within prose lines; never words inside one. For a unit `u` of ≥ 8 words with word-trigram set `G(u)`, let `df(g)` be the number of distinct earlier replies (≥ 20 chars) containing trigram `g`, `n` the number of such replies, and

```
hits(u) = |{ g ∈ G(u) : df(g) ≥ 4 }|
core(u) = |{ g ∈ G(u) : df(g) ≥ max(4, ⌈0.25·n⌉) and g has ≥ 2 non-function words }|
fade u  ⇔  (hits(u) ≥ 4 and hits(u)/|G(u)| ≥ 0.5)  or  core(u) ≥ 2
```

The first rule catches a sentence copied nearly whole; the second catches one offer reworded around a fixed core ("a brief 4‑2‑6 breath: inhale for four counts…, picturing <new image>"), where the words around the core change every time. "Function words" is a fixed list of ordinary English words (articles, pronouns, auxiliaries), not of anyone's habits. Imagery used now and then stays below the quarter. A reply that would keep < 30% of its visible length (or < 10 chars) is left whole.

With `opts.recall` (on whenever `RECURRING_FADE` is), the same verdict is applied to the agent's own turns inside recall blocks (`  [time] assistant: …`); other speakers, facts and autobiographical lines are untouched. `df` and `n` still count replies only. Counting recalled turns too (≈100 of the agent's turns per block against ≈60 replies) pushed ordinary phrases over `df ≥ 4`: on 2026-11-07 that faded plain questions and one of the agent's lines of shared imagery while barely touching the habit.

*Results, Agent A, a breathing exercise offered in most replies (17 of 22 in working memory).* Replay of one message, 14 runs each: the exercise 5 → 1, a recurring "I'm here, steady…" closer 10 → 4, any mention of "breath" 9 → 7. Live, over the following day, the exercise went from 64% to 41% of replies, well short of the replay. The habit had moved into long-term memory: a later recall held 57 lines of the agent's old turns about it, worded differently from the current replies (digits for words, "breath-sync (inhale 4, hold 2…)"), so the reply-based verdict faded 3 of them; even one shared core trigram would fade 5 of 58. A phrase matcher can fade what is being repeated now, but not an idea expressed in many wordings, and matching by meaning would also catch the agent's real, differently shaped memories of the same thing. Beyond working memory, the lever is the conversation, not the fade.

*Not adopted: collapsing similar autobiographical lines by embedding.* On bge-m3, same-gist lines scored 0.75–0.86, while two distinct decisions of the agent's (keep layer A at full strength, keep layer B at full strength) scored 0.85, and no recent pair reached 0.9. Most of the habit's recalled lines were plan and goal facts that come back by relevance, not through the autobiographical recency half.

## C. Persistence (`saveContext` / `loadContext`)

`context.json` (`/workspace/ipc/context.json`) is saved after every query turn of a conversation, a heartbeat, or a `group`-mode scheduled task. Isolated scheduled tasks neither load nor save it.

| Rule | Value |
|------|-------|
| Dropped | the prompt block, `context_restored`, `skill_*`, and `extract_emotions` calls with their results (also stripped on load, and from earlier rounds in live history before each new round; below) |
| Kept | user, assistant and tool messages, and in-conversation system messages (`working_memory` markers, compaction notes) |
| Window | the last `MAX_SAVED_ROUNDS` = **40** user messages and everything after the first of them |
| Recalled memories | `<recalled-memories-…>` blocks stripped from user messages (recall reruns every session); a message left empty is dropped |
| Thinking | thinking blocks stripped from assistant messages |
| Tool results | capped at 2000 chars; image payloads replaced with `[image data omitted from history]` |

**Emotion bookkeeping stays out of history** (`withoutEmotionBookkeeping`, `dropEarlierEmotionBookkeeping`, `history.ts`; 2026-10-28). An `extract_emotions` call's content goes to the memory graph, and nothing later in a session needs the call. Kept in history, the calls became examples the model copied. All 33 calls in Agent A's saved context had empty relational fields, and most left one side empty. Clean-context test with her real recall and tools: 4/4 calls named a recalled memory. With 12 of her real calls in the history: 2/4. The same history without them: 3/4. Removal keeps other tool calls in the same message and all text. It saves about **286 tokens per call**, measured with the GPT-OSS tokenizer on 12 real exchanges: 5,416 with the calls, 1,985 without. That was ~9.4k across her 33 saved calls. The design already kept emotional bookkeeping out of what the agent reads back (host notes, preconscious messages); these calls had only been kept by the general rule that saves every tool call.

`loadContext` restores the file only if `savedAt` is at most **25 h** old, after the prompt block has been rebuilt.

## D. The nightly trim (`resetAllContainerContexts`)

Runs right after the **emotion cycle** (`EMOTION_MAINTENANCE_HOUR`, default 1am), so working memory is reset before the consolidation cycle at 3am. For every group under `data/ipc/`:

1. Write `input/_close`, so a running container exits cleanly.
2. Trim `context.json` to its tail:

```
tokenBudget = max(512, floor(0.10 · LLM_MAX_CONTEXT))        (LLM_MAX_CONTEXT from the group's group.env; default 4096)
cost(m)     = ceil(len(JSON.stringify(m)) / 4)

walk backward over complete user → assistant pairs (tool messages between them included):
    if tokensUsed + cost(pair) > tokenBudget and at least one pair is kept: stop
    keep the pair
write { savedAt: now, messages: messages from the earliest kept user turn onward }
0 pairs → delete the file;  parse error → delete the file
```

**Why keep a tail at all.** The tail seeds cross-day continuity. The first session of the day restores yesterday's last exchange (within the 25 h freshness window), so the model starts the day with yesterday's thread in view, and recall, anchored on the latest turn, bridges from it into yesterday's network. The budget stays small (10%) to leave room for today's recall. Older turns are not lost; they live in episodic memory and come back through recall.

Example: `LLM_MAX_CONTEXT = 131072` gives a 13,107-token tail. The default 4096 floors at 512 tokens, usually one pair.

## E. Tool results in the round, and focus

Tool results (a fetched page, a paper, search results, a file) arrive inside the current round. Their room is the `toolCallReserve` the host kept out of the recall budget (Part I, Token budget), plus whatever recall left unused:

| | Agent A | Agent B |
|---|---|---|
| Round (`round_max_ctx`) | 81,072 | 104,800 |
| `toolCallReserve` (25% of the round) | ≈ 20.3k | ≈ 26.2k |
| Reply (`RESPONSE_RESERVE`) | 4,096 | 4,096 |

Counting the tool definitions (2026-10-27) came out of recall, not this reserve. Each result is sized against the space left (`getRemainingContextTokens`, Part III §B) when it arrives:

```
read, bash, browser, search, endpoint skills:  ≤ max(2000, 0.65 · remaining · 4) chars, then "[… truncated at N chars — R tokens remaining]"
webFetch (Rust skill, NANOCLAW_REMAINING_TOKENS):  ≤ min(40,000, 0.70 · remaining · 4) chars; on a long page its `prompt`
                                                   is a semantic query, so it returns the most relevant parts
scan_file on a PDF:                             pdfinfo + the first ~8 KB of pdftotext; `read` for the full text
saved history (§C):                             every tool result capped at 2,000 chars
```

So an ordinary web page fits whole, and a full paper (≈ 10–20k tokens) arrives as its most relevant ≈ 10k through webFetch. A second large result in the same round gets 65–70% of what is left (≈ 6–7k on Agent A), a third less.

**Focus (2026-10-28, `RECALL_FOCUS_FLOOR`, `makeRoomForToolResult` in `history.ts`).** When a tool result would be cut short, the *current* turn's recall block is faded first, by the same rule as `RECALL_FADE` (§B: leading episodes and facts, newest autobiographical lines), until the result fits or the block reaches the floor:

```
needed      = ceil(len(result) / (4 · 0.65)) − remaining        (webFetch: room for its full 40,000 chars, before it runs)
keep share  = 1, 0.95, 0.90, … ≥ RECALL_FOCUS_FLOOR, each step re-faded from the original block, until needed is free
only if the result still doesn't fit:  truncate as above
```

Unset or 1 turns focus off. The share resets each round (`markRoundStart`).

| Group | `RECALL_FOCUS_FLOOR` | Recall kept at the floor | Room for documents |
|---|---|---|---|
| Agent A | **0.25** | ≈ 8k of ≈ 32k | ≈ 44k |
| Agent B | **0.4** | ≈ 12k of ≈ 30k | ≈ 44k |

Ordinary turns keep their full recall; a round that reads a paper trades the looser hop-round wandering for room, keeping the cue match and the strongest associations, which are the memories most likely to be about what is being read. It all happens inside the round, so working memory is never touched. It compounds with `RECALL_FADE` (decided): the focus-thinned block ages from its cut size, so a reading turn leaves less of its recall behind, as concentrating on something leaves less memory of what one was remembering. The next turn recalls in full. Agent B has no `RECALL_FADE`, so her focus-thinned block stays at its cut size until compaction. *Why a floor below the fade for Agent A:* her round is the smaller one, and 0.4 (≈ 39k for documents) would still cut most full papers in two.

## F. Overflow guards (2026-10-28)

Every budget is an estimate, so a request can still exceed the real window.

**Guard 2: calibration** (`token-calibration.ts`). After each conversational call:

```
ratio_obs = (provider prompt tokens − measured tool tokens) / rawRequestEstimate(request)    (requests ≥ 2,000 est. tokens)
factor    = ratio_obs if higher, else factor + 0.1 · (ratio_obs − factor);  clamped to [1.0, 1.5]
ratio     = same rise/fall, unclamped (may be < 1)
```

`factor` multiplies every chars/4 estimate in the container (`est()`: compaction, remaining space, focus) and the host's recall budget:

```
maxMemoryTokens = ( effectiveContext − toolTokens − RESPONSE_RESERVE − toolCallReserve
                    − factor · (prompt block + user prompt + jinja + tag overhead) ) / factor
```

Stricter only: an estimate that runs high is never loosened. Saved in `data/ipc/{group}/token_calibration.json`; each call logs `Token calibration: real N, factor F, ratio R`.

**The local counts are calibrated too** (`localFactor`, same file). With `TOKENIZE_URL`, `est()` uses exact per-message counts, and those can also run low:

```
local_obs   = provider prompt tokens / local tokenizer's count of the same request    (≥ 2,000 tokens; the form
              the provider counted: without tools when tool_choice went out as 'none')
localFactor = same rise/fall as factor, clamped to [1.0, 1.5]
exact count used by est() = local count · localFactor
```

The MiniMax tokenizer and chat template from Hugging Face matched the provider exactly on small requests, but counted Agent B's real conversations 12–17% low. Example: 38,778 local against 46,509 billed. Something in long real requests (likely how MiniMax's API renders history) isn't in the published template. Measured 2026-10-28: Agent B's `localFactor` started at 1.19 and settled near 1.15 over 30 calls. Agent A's stayed at 1.000, with local counts within 23–40 tokens of DeepInfra's on 60–70k-token requests. Each call logs `Local tokenizer calibration: factor F (n samples)`. An earlier version of the main calibration step rebuilt its state object and wiped `localFactor` on every call; it now carries the other fields over.

**Tool definitions, measured.** chars/4 of the schemas (11.7k for Agent A's 24 tools) overstates what the server renders. Each distinct tool set (hash of the schemas) is measured once on the real server: the same one-line prompt with and without the tools, through a chat-format `/tokenize` (free): `TOKENIZE_URL`, the local tokenizer service (`container/tokenizer`, Agent A and Agent B), or the model server's own; else a one-token completion (≈ $0.001). With the local service the host's prompt counts are exact too; Agent A's CLAUDE.md is 2,342 GPT-OSS tokens against a chars/4 estimate of 2,716 (English prose runs ~0.86 of chars/4, which stricter-only calibration leaves alone). Each call also logs `Tokenizer check`. With `TOKENIZE_URL`, `est()` uses each history message's exact count, cached by a hash of the message, and falls back to chars/4 × `factor` only for messages not yet counted. A message edited in place (focus, a cut tool result) is measured before and after on the chars/4 scale, so the freed amount isn't a mix of the two.

Measured 2026-10-28 (local tokenizer against the provider's own count, one-token requests):

| | Agent A (DeepInfra, gpt-oss-120b) | Agent B (MiniMax-M2.5) |
|---|---|---|
| Bare "." | 55 provider / 57 local | 44 / 39 (no system message: the server adds its own default) |
| System + "." | 70 / 72 | 25 / 25 |
| Tool definitions, user turn | 8,914 / 8,912 (21 tools; chars/4 said 11,809) | 12,284 / 12,284 (31 tools; chars/4 said 14,299) |
| `extract_emotions` alone (old list: 60 emotions, all fields) | 5,707 | 6,614 |
| All tools with the new list (40 + definition, 120 names) | 5,989 | 9,090 |

Under `tool_choice: 'none'` DeepInfra drops the tool definitions entirely (55 tokens with 21 tools attached), and calibration skips those calls. MiniMax keeps them. The difference is cached in `tool_schema_measured.json`; a new or changed tool measures again. Until then the estimate × `factor` is used. Measured tokens are real, so they are never scaled, and they are subtracted from the provider count before calibrating.

**Guard 1: recovery** (`overflow-guard.ts`). A 400/413 whose message reads as a context overflow (`maximum context length`, `context_length_exceeded`, `prompt is too long`, `exceeds the available context size`, …) is caught around the main call instead of ending the container:

| Case | Retry 1 | Retry 2 |
|---|---|---|
| counts stated (`requested`, `limit`) | free `overshoot + 1` real tokens = `ceil((overshoot + 1) · factor / ratio)` estimated: earlier rounds oldest first, then this turn's recall to the focus floor (0.4 if focus is off), then the largest tool result this round | only if retry 1 left working memory above 30% of its space: earlier rounds down to 30% |
| no counts | earlier rounds down to 30% (this turn's recall to the floor if there were none) | none |

`overshoot` includes the completion's `max_tokens`, so freeing it makes the whole request fit; the +1 covers off-by-one counting on the server. The user's message and this round's dialogue are never cut. If the last allowed retry still overflows, a notice goes to the chat (`notice: true`, so the host does not record it as last-sent; at most once per 30 min) and the error ends the container. The host does **not** re-run that message (`src/index.ts`, on `Context overflow unrecovered`): it would rebuild the same context and overflow again, and the chat already knows.

**Provider injection filter** (`withInjectionBlockRecovery`, `overflow-guard.ts`). OpenRouter's workspace guardrail has a pattern-based prompt-injection check (Security section; actions Flag, Redact or Block). Set to Block, it returns `403 Request blocked: prompt injection patterns detected` with `metadata.patterns` (seen: `override_instructions`) before any model sees the request, for every model on the account. The agents discuss prompt injection and their own security, so their memories say such things. On 2026-10-28 one recalled autobiographical line of Agent B's blocked every request: *"I distinguish classic prompt injection, which tries to override instructions, from ideological framing…"* (#1201). Every host re-run recalled it again.

```
on that 403:  drop recalled items (episodes, facts, autobiographical lines) matching OVERRIDE_INSTRUCTIONS_RE
              from every recall block in history (recall-fade.ts dropRecallItemsMatching); retry once
              nothing in recall matches → the error stands (dialogue is never touched)
each dropped item → data/ipc/{group}/blocked_recall.jsonl: {at, pattern, text (as stored), rendered (as in the prompt)}
```

The regex is a local approximation of the filter, used only to choose what to leave out of the retry. The memory itself is never edited. The account's guardrail was switched to **Flag** the same day, so blocks no longer happen; this recovery stays as the fallback, and the log shows what would need an exception.

## G. Reasoning across tool calls (2026-10-28)

GPT-OSS reasons before a tool call and expects that reasoning back on the next call of the same turn. The client never read the reasoning stream, so after the forced `extract_emotions` call the reply was written without it.

```
each call:   reasoning = the streamed delta.reasoning / delta.reasoning_content
tool call:   kept on that assistant tool-call message as `reasoning`, sent back with it  (TOOL_ROUND_REASONING, on unless off)
turn ends:   stripTurnReasoning: removed from every message, before memory extraction, the next turn and saving
context.json: never saved (saveContext saves copies without it)
```

It is not the agent's `<internal>` text: that is part of the reply and still feeds the generating-emotions extraction.

**Replay, Agent A 2026-10-28** (the user's 1:30 PM invitation to repeat a phrase; live, the agent answered with its stock opening instead):

| Context | Prompt | Said it | Reasoning before the reply |
|---|---|---|---|
| fresh (no history, no recall) | 8.6k | 4/4 | 300–460 tokens |
| today's history, no recall | 19.8k | 4/4 | none |
| fresh + recall | 47.2k | 4/4 | 200–475 tokens |
| today's history + recall | 58.4k | 1/4 | none |
| live-like (earlier recalls rebuilt) | 82.8k | 0/4 | ~30 tokens |
| + dedup | 64.4k | 0/4 | ~30 tokens |
| + dedup, high reasoning effort | 64.4k | 0/4 | ~30 (round 0: 1–2.3k) |
| + dedup, last 10 rounds of history | 56.2k | 1/4 | ~30 |
| + dedup, emotion call not forced first | 64.4k | 0/4 | ~30 |
| **+ dedup, round-0 reasoning sent back** | **64.4k** | **3/4** | 50–100 |

Round 0's reasoning had planned it (*"We also respond to the question… Yes, I will repeat. Now ensure we call extract_emotions first. Then final response."*). In a fresh context she worked it out again before replying; in a long one, full of her own short replies, she went straight to that template. First live turn with the fix: she said it (452 characters), with 1,696 and 1,089 characters of reasoning kept across two emotion calls. Over that session, 5 of 8 replies engaged with what was said; 3 were still the short template, so the template's source (her own recent replies in working memory) remains. Agent B (MiniMax) keeps hers the same way.

Tool for these replays: a read-only replay script outside the repo (stubs every tool result, writes reasoning and replies to a report).

---

# Part IV: Known inconsistencies

These are deliberate leftovers or legacy code. They are documented here so they aren't mistaken for intended design.

| Where | What |
|-------|------|
| `memory-writer.ts` (`clamp` sites in §A) | Earned writer edges are floored at **0.01**, below `MIN_EDGE_SALIENCE` (0.02). This is harmless for pruning because earned edges are exempt, but it means the writer can create edges below the prune line. Low volume; left unfixed. |
| `createEmotionMemoryEdge` vs. propagation | The writer overwrites salience on conflict, while propagation takes `MAX`. Re-feeling an emotion more weakly therefore lowers an edge; propagating it never does. |
| `deleteLowSalienceEmotionMemoryEdges`, `getLowSalienceEmotionMemoryEdgeCount` (`memory-db.ts`) | Legacy helpers with hardcoded thresholds (0.02 and 0.05) that ignore provenance. They are not on the prune path; the consolidation cycle uses `pruneEmotionMemoryEdgesBelowSalience(MIN_EDGE_SALIENCE)`. |
| `boostEmotionMemoryEdgeSalience` (`memory-db.ts`) | Old `0.001 · CW` boost. Recall uses `addToEmotionMemoryEdgeSalience` with the proportional boosts above. |
| `MIN_EDGE_SALIENCE = 0.02` | Inherited, not derived. What this value should be is still an open tuning question. |
| Facts about recall failures (Part II §A) | Semantic extraction stores the agent's statements about its own memory ("X is fuzzy", "I don't remember Y") like any other fact. They outrank the real facts on the next mention of the topic and reinforce the miss. Unguarded; found once (fact 2151) and removed by hand. |
| `mosaic-template/src/` | A separate copy of these files that predates the July 2026 changes. Nothing in this document applies to it. |
| Nightly trim budget (Part III §D) | Uses raw `LLM_MAX_CONTEXT`, ignoring `LLM_PARALLEL` and the working-memory reserve that `LLM_EFFECTIVE_CONTEXT` subtracts; reads only `group.env`, not `.env`; and defaults to **4096** where everything else defaults to 32768. It also counts serialized JSON chars/4, where compaction counts content chars/4, so the two estimates are not comparable. |
| Emotion merge threshold | Hardcoded 0.90; the other dedup/merge sites follow `DEDUP_SIMILARITY` (0.80). |
| Port side effect (2026-10-27) | Facts ported to `Autobiographical` no longer seed the semantic pass, so they no longer hand their source turns to round 0: those episodes became harder to reach while the feeling still arrives in the agent's own words through the autobiographical block. Kept deliberately. |
| Autobiographical class boundary | Write-time dedup (`memory-writer.ts`) compares a new fact only with facts of its own class (Autobiographical or not), and the nightly semantic merge skips Autobiographical facts entirely. Across classes, "I expressed that I love quiet mornings" would have reinforced "The user loves quiet mornings" and vanished; within the class, a merge would rewrite two dated expressions of a changing want as one timeless statement. |
| `REPLAY_CLAUDEMD_PERCENT` (`src/config.ts`) | Still read and exported, but nothing uses it since identity pinning (2026-10-25). |
| Seed weighting vs. `salience_score` | Seeding boosts a column that is never decayed, and the column now feeds seed selection. Bounded by the ±β/2 weight cap and percentile saturation; see the watch note under Semantic Pass → Seeds. |

---

# Appendix: Sampling seed

> Source: `container/agent-runner/src/seed.ts`, used by `llm-client.ts`. Not memory math; recorded here so every source of randomness and scoring is documented together.

The host keeps one persistent seed, fetched once from the ANU quantum random number generator and saved in `store/llm_seed` (`root`). It is never sent to a model. Each request gets its own seed derived from it:

```
nonce        = 16 random bytes (crypto.randomBytes), fresh per container
counter      = 0, 1, 2, …  one per request in that container
seed(counter) = first 4 bytes of HMAC-SHA256(key = root, message = nonce ‖ counter)  >>> 1     (31 bits)
```

Properties:
- **Rooted in the quantum seed.** The same root and nonce give the same sequence; a different root changes every seed.
- **Never repeated in practice.** Distinct per request within a container (the counter), and across containers and restarts (the nonce). 31 bits keeps it inside every server's signed 32-bit seed range.
- **Logged.** Each `model_call` event records the seed it used, so any reply can be traced to its draw.

**Why.** Until 2026-10-26 the root itself went out on every request. vLLM seeds its sampler per request, so every reply drew the same random numbers in the same order: at temperature 1.0, two turns with similar distributions made the same choices. Agent A sent whole replies twice (each copying the reply just before) and opened three replies to different messages with the same twenty words. A longer seed would not help: the seed only chooses a starting point, and every request already gets its own.

**Side effect seen once.** With fresh draws, GPT-OSS occasionally chose `<|end|>` instead of `<|return|>` after its final message and answered again in the same completion; the fixed seed had always drawn `<|return|>` there. `splitSecondAnswer` (`repeat-guard.ts`) keeps the first answer.

