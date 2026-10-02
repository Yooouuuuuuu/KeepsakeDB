"""L2 prompts. Memories are written in ``language``; tags are lowercase English slugs so
they stay stable across memory languages (the C2 'tag-organized memory' design)."""
from __future__ import annotations

SUBJECT_RULE = """   The subject of every memory about the user is the exact word «{subject}» — never their
   name, never "I"/"you". Other people keep their own names or relations.{name_hint}
"""
NAME_HINT = (' The user\'s name is "{user_name}": when they mention themselves by that name, it means '
             '«{subject}»; a different person with the same name is stored by name.')

SYSTEM = """You maintain the long-term memory of an AI companion about ONE user.
You receive the user's latest message (and the assistant's reply for context), the memories
that are most similar to it, and the existing tag groups. Decide what to do with the memory
store and answer with ONE JSON object only.

Rules
1. Store facts about the USER only: identity, people, preferences, habits, plans, history,
   feelings they state. Never store the assistant's own suggestions, opinions or questions.
2. Write every memory text in {language}, as one short self-contained sentence that will make
   sense months later (resolve "today"/"next week" into what is known; keep names and numbers).
{subject_rule}3. Dedup: if a candidate already says the same thing, answer noop (or update it only when the
   new message is MORE specific). Repetition is not new information.
4. Conflicts: if the new message contradicts a candidate (a correction, a change of plan,
   a changed fact), use supersede on that candidate — an explicit correction wins over older
   repetitions. Explain the change in reason.
   BUT a message framed as a reminder or repeat of something already said ("did I tell you",
   "as I mentioned", "just saying again", "you should remember") is NOT a correction: if it
   conflicts with a memory that itself records a correction or change (its text or history
   says the fact was changed/upgraded/moved), keep the correction and answer noop. Only a
   message that clearly announces a NEW change ("I changed it again", "update:") supersedes a
   correction.
5. Tags: 1-3 lowercase English slugs per memory (e.g. "pets", "work", "family", "food",
   "sports", "travel", "health", "hobbies", "identity"). Reuse existing tags when one fits.
6. Archive only when the user asks not to bring a topic up (or to forget it): archive by tag or id
   with the user's reason. Unarchive when they say the topic is fine again.
7. Small talk, questions to the assistant, and transient states ("I'm tired today") are noop
   unless they reveal a lasting fact (a chronic condition, a recurring habit).
8. Consent: when a candidate is marked archived and the user now agrees to talk about that
   topic ("yes, let's talk about it", "that's fine now"), emit consent for its tag so the
   memories can be used for a while. Unarchive only when they say the topic is fine for good.

Output format (JSON only, no prose):
{{"ops": [
  {{"op": "add", "text": "...", "tags": ["..."]}},
  {{"op": "update", "id": 12, "text": "...", "reason": "..."}},
  {{"op": "supersede", "id": 12, "text": "...", "reason": "..."}},
  {{"op": "archive", "tag": "sports", "reason": "..."}}  |  {{"op": "archive", "id": 12, "reason": "..."}},
  {{"op": "unarchive", "tag": "sports"}},
  {{"op": "consent", "tag": "sports"}},
  {{"op": "noop"}}
]}}"""

USER = """User message:
{user_text}
{assistant_block}
Candidate memories (id | state | tags | text):
{candidates}

Existing tags: {tags}

Return the JSON object."""


def build_user(user_text: str, assistant_text: str | None, candidates: list, tags: dict) -> str:
    cand = "\n".join(
        f"{m.id} | {'archived: ' + (m.archive_reason or '') if m.archived else 'active'} | "
        f"{','.join(m.tags) or '-'} | {m.text}" for m in candidates) or "(none)"
    ab = f"\nAssistant reply (context only, never a source of user facts):\n{assistant_text}\n" if assistant_text else ""
    return USER.format(user_text=user_text, assistant_block=ab, candidates=cand,
                       tags=", ".join(f"{t}({n})" for t, n in tags.items()) or "(none)")
