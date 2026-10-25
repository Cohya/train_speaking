from __future__ import annotations

import pytest

from train_speaking.voice.sentence_splitter import limit_sentences, split_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("   ", []),
        ("Hello there, welcome.", ["Hello there, welcome."]),
        (
            "Thanks for joining. Tell me about yourself! What drew you here?",
            ["Thanks for joining.", "Tell me about yourself!", "What drew you here?"],
        ),
        (
            "We use tools, e.g. Python and SQL. Version 3.5 is fine.",
            ["We use tools, e.g. Python and SQL.", "Version 3.5 is fine."],
        ),
        ("Okay. Thanks. Now, describe a conflict.", ["Okay. Thanks.", "Now, describe a conflict."]),
        ('He said "stop." Then we left.', ['He said "stop."', "Then we left."]),
        ("No final punctuation", ["No final punctuation"]),
        ("Okay.", ["Okay."]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_whitespace_is_normalized() -> None:
    assert split_sentences("First   sentence here.\n\nSecond one here.") == [
        "First sentence here.",
        "Second one here.",
    ]


def test_limit_sentences() -> None:
    text = "One. Two. Three. Four."
    assert limit_sentences(text, 3) == "One. Two. Three."
    assert limit_sentences("Only one", 3) == "Only one"
