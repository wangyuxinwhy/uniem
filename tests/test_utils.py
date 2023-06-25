from uniem.utils import convert_to_readable_string


def test_convert_to_readable_string():
    assert convert_to_readable_string(123) == "123"
    assert convert_to_readable_string(1234) == "1.2k"
    assert convert_to_readable_string(1234567) == "1.2M"
    assert convert_to_readable_string(1234567890) == "1.2B"
