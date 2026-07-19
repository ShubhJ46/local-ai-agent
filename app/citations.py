"""Check that the files an answer cites were actually shown to the model.

Retrieval can be strong and the answer still wrong: a model asked about an
endpoint will happily name a plausible path it never saw. Grounding the prompt
reduces this but cannot guarantee it, and the guarantee is the point — a code
assistant that invents file names is worse than one that says it does not know,
because the invented name looks exactly like a real answer.

This is deliberately deterministic and offline: no second model call to check
the first, so it runs in CI and cannot itself hallucinate.
"""

import re

from app.loader import SUPPORTED_EXTENSIONS

# Only tokens carrying an extension we actually index are treated as citations.
# Bare identifiers are excluded on purpose: "the OwnerController class" is prose,
# and trying to verify it produces false alarms on ordinary English. Requiring a
# known extension also keeps version numbers like "Spring 3.2" from matching.
_EXTENSIONS = "|".join(extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS)
FILE_TOKEN = re.compile(rf"[\w./\\-]*\w\.(?:{_EXTENSIONS})\b", re.IGNORECASE)


def _basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def cited_files(text: str) -> set[str]:
    """Return the file names a piece of text refers to, without directories."""
    return {_basename(match.group(0)) for match in FILE_TOKEN.finditer(text or "")}


def unverified_citations(answer: str, available: set[str]) -> list[str]:
    """Return cited file names that were never shown, in a stable order."""
    return sorted(cited_files(answer) - {_basename(name) for name in available if name})


def verify(answer: str, available: set[str]) -> str:
    """Append a correction when the answer cites something it was not shown.

    The unverified name is flagged rather than deleted. Removing it would leave
    fluent prose making an unsupported claim, which is the failure mode this is
    meant to expose.
    """
    unverified = unverified_citations(answer, available)
    if not unverified:
        return answer

    known = sorted({_basename(name) for name in available if name})
    sources = ", ".join(known) if known else "nothing"

    return (
        f"{answer}\n\n"
        f"[unverified: {', '.join(unverified)} "
        f"{'was' if len(unverified) == 1 else 'were'} not in the retrieved source. "
        f"Retrieved: {sources}.]"
    )
