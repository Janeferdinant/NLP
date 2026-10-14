"""Quoted dataset messages in the error analysis are shown exactly as they appear in the public dataset."""

from spam_classifier.report import quote_message


def test_phone_numbers_and_names_are_not_masked():
    text = "Hi I'm Claire, call 08001234567 or 07123 456789 now"
    assert text in quote_message(text)


def test_long_messages_are_not_truncated():
    text = "word " * 120 + "end"
    assert quote_message(text).strip("` ").endswith("end")
    assert len(quote_message(text)) >= len(text)


def test_line_breaks_become_spaces_so_each_message_stays_on_one_line():
    assert "\n" not in quote_message("first line\nsecond line\r\nthird")


def test_backticks_in_a_message_survive_inside_the_code_span():
    quoted = quote_message("use `code` and ``more`` here")
    assert quoted.startswith("```") and quoted.endswith("```")
    assert "use `code` and ``more`` here" in quoted
