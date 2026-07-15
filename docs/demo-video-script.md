# Demo Video Script — "RAG: Naive → Production-Grade"

Target length: ~5:00. Part 1 is verbatim (already recorded/sourced). Part 2 is new —
written to be spoken, not read; short sentences, direct address, one idea per beat.
Timestamps are estimates for rehearsal pacing only.

---

## Part 1 — Naive RAG (verbatim, 00:00–01:02)

**[00:00:00]**
RAG is one of the most important skills for AI engineers to have right now. Here's
exactly how it works under the hood and free resources to learn. Step one, chunking.
Take your documents, which could be PDFs, internal wikis, web pages,

**[00:00:11]**
Slack messages, really anything you want, then split them into chunks. But not
randomly, you want to break them at semantic boundaries. Think paragraphs, sections,
headers, and you'll leave overlap between these chunks, so that way when you retrieve
one of them, you don't lose the surrounding context.

**[00:00:26]**
Step two, embedding. Run each chunk through an embedding model. The embedding model
will convert the text into high dimensional vectors, AKA a list of numbers that
represent the meaning of that text. Chunks that have similar meaning will be closer
together as vectors.

**[00:00:39]**
Step three is storage. You want to store these vectors in a vector database like
Pinecone, Weaviate, and PG Vector. And step four is the retrieval and generation.
When a user asks your system a question,

**[00:00:49]**
you embed that question the same way you did the other chunks. Then you look for the
closest matching vectors to their question, pull the original text from those chunks,
and feed it to an LLM as context. Now the LLM model will answer with relevant context
rather than just training data.

**[00:01:02]**
This might have sounded complex, but it's actually pretty simple. Production level RAG
is a completely different beast, and it gets a lot deeper than this. I suggest you
check out these free resources to learn more about RAG. Comment search for the links.

---

## Part 2 — What "production" actually means (new, ~01:02–05:00)

**[00:01:02] Transition**
That's naive RAG. And it works — until someone asks a question your documents don't
actually answer. Naive RAG can't tell "I found something relevant" from "I found the
right answer." Here's everything I actually changed, building a RAG system over SEC
financial filings.

**[00:01:16] Chunking, revisited**
Start with chunking again, because "split at semantic boundaries" hides a real
decision. Fixed-size splitting cuts blindly — mid-sentence, mid-table, doesn't care.
Recursive splitting respects structure — paragraphs first, then sentences, only
falling back to a hard cut when it has to. Token-based splitting sizes chunks to what
the embedding model actually sees, not to a character count that may not match. And
for something like a balance sheet, none of those are really enough — tables need
their own strategy. More on that in a minute.

**[00:01:44] Refusal-first — the core differentiator**
Refusal-first design. Two separate gates where the system is allowed to say
"I don't know" instead of guessing. Gate one, before any retrieval: does the company
and fiscal year actually exist in the corpus? Wrong year, wrong company — refuse
immediately, zero retrieval wasted. Gate two, after retrieval: a cross-encoder scores
how relevant the best chunk really is. Below threshold, refuse instead of generating
from weak context.

**[00:02:08] Single source of truth**
Second lesson: never denormalize. Company name, fiscal year, currency — those live in
exactly one place, a metadata file, resolved at query time. Not copied onto every
chunk. Fix a company's fiscal year once, not across ten thousand chunks.

**[00:02:24] Hybrid retrieval + reranking**
Hybrid retrieval. Pure vector search misses exact terms — company names, tickers,
line items like "shareholders' equity." So retrieval combines BM25 keyword search
with vector search, then reranks the combined candidates with a cross-encoder.

**[00:02:40] Security**
Security isn't just "add a regex and call it done." User input is never formatted
directly into the prompt — it's bound as a separate variable, so there's no template
string to inject into in the first place. PII — emails, SSNs, card numbers — gets
masked before it ever reaches the model.

**[00:02:56] Reliability**
Every model call can fail. So there's retry with backoff at the provider level, a
second retry wrapper around the whole request, and a fallback to a stronger model if
the primary keeps failing.

**[00:03:08] Caching**
Ask the same question with the same settings, get an instant cached answer. The cache
key includes the mode and any overrides — so a test run with different settings can
never accidentally serve someone else's cached result.

**[00:03:20] Token budget**
There's a real token budget too — not on the question you type, on the assembled
prompt the model actually sees. Retrieved context gets trimmed to fit, dropping the
lowest-ranked chunks first, never the question itself.

**[00:03:34] Observability — three lenses**
Three separate lenses on what's happening. Traces show the exact path one question
took — parse, resolve, retrieve, rerank, generate, step by step. Metrics show system
health over time — latency, error rate, cache hit rate. Logs are structured JSON, one
line per request, built for aggregation, not for someone squinting at a terminal.

**[00:03:56] A real, still-open lesson**
Here's something I didn't expect. Dense financial tables — the exact numbers people
actually ask about — often rank *worse* than the prose around them in vector search.
A table is mostly numbers and pipe characters; it doesn't "read" semantically the way
a sentence does. That's a real, still-open problem in RAG over structured documents.

**[00:04:14] What I built and then removed**
I also built an agentic retrieval loop — LangGraph, a self-correcting retry step, the
whole thing. Then I looked closely at what the retry step actually did. It didn't
rewrite the query at all — it just retried the identical search. It wasn't earning
its complexity. So I cut it. A production codebase isn't measured by what you can
build. It's measured by what you can justify keeping.

**[00:04:34] Evaluation — the part that makes any of this credible**
None of it means anything without measurement. The system is graded against a
40-question adversarial benchmark — a chunk of those questions are deliberately
unanswerable, on purpose. The headline metric isn't overall accuracy, it's
correct-refusal rate on the unanswerable ones — because a system that just says
"I don't know" to everything would score great on raw accuracy and be useless.
Right now: 100% correct refusal on the adversarial set, 85 to 88% overall.

**[00:04:56] Close**
Chunk, embed, retrieve, generate — that's maybe 20% of the work. The other 80% is
refusing correctly, retrieving precisely, staying reliable when things fail, knowing
what to cut, and proving all of it with real numbers, not vibes.

**[00:05:00] End**
