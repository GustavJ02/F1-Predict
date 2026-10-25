from f1live.paths import Session, folder_name


def test_folder_name_has_no_spaces_or_accents():
    assert folder_name("United States") == "United_States"
    assert folder_name("São Paulo") == "Sao_Paulo"
    assert folder_name("Saudi_Arabian") == "Saudi_Arabian"
    assert folder_name(" Abu  Dhabi ") == "Abu_Dhabi"


def test_session_accepts_either_spelling():
    a, b = Session("2025", "United States", "quali"), Session.parse("2025/United_States/quali")
    assert a == b and a.dir.name == "quali" and a.dir.parent.name == "United_States"
