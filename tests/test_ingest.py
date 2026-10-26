from app.ingest import chunk_document, clean_text, extract_pages, find_section


def test_clean_text_joins_hyphenated_words_and_collapses_spaces():
    assert clean_text("tempera-\nture   compensated\n\n\n\nok") == "temperature compensated\n\nok"


def test_find_section_detects_numbered_headings():
    text = "intro line\n4.1 Zero adjustment\nVent the transmitter"
    assert find_section(text, "Unknown") == "4.1 Zero adjustment"
    assert find_section("no heading here", "3 Installation") == "3 Installation"


def test_extract_pages_returns_one_entry_per_page(manual_bytes):
    pages = extract_pages(manual_bytes)
    assert len(pages) == 6
    assert "PT-200" in pages[0]


def test_chunks_carry_page_and_section_metadata(manual_bytes):
    chunks = chunk_document(manual_bytes, "manual.pdf", chunk_size=300, chunk_overlap=30)
    assert all(len(c.text) <= 300 for c in chunks)
    assert {c.page for c in chunks} == {1, 2, 3, 4, 5, 6}
    torque = next(c for c in chunks if "25 Nm" in c.text)
    assert torque.page == 3
    assert torque.section.startswith("3.1")
    assert len({c.id for c in chunks}) == len(chunks)  # ids are unique


def test_smaller_chunk_size_produces_more_chunks(manual_bytes):
    small = chunk_document(manual_bytes, "m.pdf", chunk_size=256, chunk_overlap=32)
    large = chunk_document(manual_bytes, "m.pdf", chunk_size=1024, chunk_overlap=128)
    assert len(small) > len(large)
