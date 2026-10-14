from spam_classifier.preprocessing import normalize_text


def test_lowercases():
    assert normalize_text("FREE Entry NOW") == "free entry now"


def test_replaces_urls_with_a_token():
    out = normalize_text("Visit http://win.example/prize or www.claim.example today")
    assert "http" not in out and "www" not in out
    assert out.count("urltoken") == 2


def test_replaces_long_digit_runs_but_keeps_short_numbers():
    out = normalize_text("call 08001234567 now, 2 for 1 offer")
    assert "08001234567" not in out
    assert "phonetoken" in out
    assert "2 for 1" in out


def test_replaces_currency_amounts():
    out = normalize_text("You won \xa31000 or $50")
    assert out.count("moneytoken") == 2
    assert "1000" not in out


def test_collapses_whitespace():
    assert normalize_text("a   b\n\n c") == "a b c"


def test_is_idempotent():
    once = normalize_text("WIN \xa3500 at http://x.example call 07123456789")
    assert normalize_text(once) == once
