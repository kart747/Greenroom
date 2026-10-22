from coach.text_stream import SentenceChunker


def chunk(tokens):
    c = SentenceChunker()
    out = [s for t in tokens for s in c.feed(t)]
    if rest := c.flush():
        out.append(rest)
    return out


def test_splits_on_sentence_ends_as_tokens_arrive():
    c = SentenceChunker()
    assert c.feed("Hello there") == []
    assert c.feed("! How are") == ["Hello there!"]
    assert c.feed(" you? Fine.") == ["How are you?"]
    assert c.flush() == "Fine."


def test_does_not_split_titles_or_eg():
    assert chunk(["Mr. Rao will see you. Dr. Shah is ", "away, e.g. on leave. Okay?"]) == [
        "Mr. Rao will see you.", "Dr. Shah is away, e.g. on leave.", "Okay?"]


def test_decimals_survive_token_boundaries():
    # "3." arrives first; only the next token shows it is a decimal.
    assert chunk(["It costs 3.", "5 lakh. Is that okay?"]) == ["It costs 3.5 lakh.", "Is that okay?"]


def test_quotes_and_newlines():
    assert chunk(['She said "yes." Then', " we left\nGood"]) == ['She said "yes."', "Then we left", "Good"]


def test_initials_and_pronoun_i():
    assert chunk(["J. K. Rowling wrote it. So did I. Really"]) == [
        "J. K. Rowling wrote it.", "So did I.", "Really"]


def test_flush_on_empty_returns_none():
    c = SentenceChunker()
    assert c.flush() is None
    c.feed("Done. ")
    assert c.flush() is None
