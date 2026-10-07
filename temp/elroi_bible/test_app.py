"""Tests run against the real built database (data/elroi_bible.db). Skipped if it has not been built."""
import os, shutil, sqlite3, subprocess, sys, unicodedata
import pytest

import app as appmod
from books import parse_reference

HERE = os.path.dirname(os.path.abspath(__file__))
pytestmark = pytest.mark.skipif(not os.path.exists(appmod.DB_PATH), reason="run build_db.py first")


@pytest.fixture()
def client():
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_reference_parsing():
    assert parse_reference("John 3:16") == ("Jhn", 3, 16, 16)
    assert parse_reference("jn 3:16-18") == ("Jhn", 3, 16, 18)
    assert parse_reference("1 Cor 13") == ("1Co", 13, None, None)
    assert parse_reference("first john 4:8") == ("1Jn", 4, 8, 8)
    assert parse_reference("Jude 3") == ("Jud", 1, 3, 3)          # one-chapter book: 3 is the verse
    assert parse_reference("Song of Solomon 2:1") == ("Sng", 2, 1, 1)
    assert parse_reference("G3056") is None


def test_home_and_redirects(client):
    r = client.get("/")
    assert r.status_code == 200 and b"ElRoi Bible" in r.data
    assert client.get("/go?q=John 3:16").headers["Location"] == "/read/Jhn/3#v16"
    assert client.get("/go?q=jn 3:16-18").headers["Location"] == "/read/Jhn/3?hl=16-18#v16"
    assert client.get("/go?q=G3056").headers["Location"].startswith("/search?q=G3056")
    assert client.get("/go?q=").status_code == 302


def test_hebrew_chapter_renders_rtl_and_untouched(client):
    r = client.get("/read/Gen/1")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and 'dir="rtl"' in html
    assert html.count('class="verse') == 31
    con = sqlite3.connect(appmod.DB_PATH)
    surface = con.execute("SELECT surface FROM words WHERE lang='heb' AND ref='Gen.1.1' AND pos=1").fetchone()[0]
    assert surface in html                     # exactly as stored: no Unicode normalization applied
    # STEP's mark order is consonant, dagesh, vowel (bet, dagesh, sheva). A typed literal can differ,
    # which is why the app never normalizes Hebrew.
    assert surface.startswith("\u05d1\u05bc\u05b0") and surface.endswith("\u05ea")


def test_greek_chapter_uses_textus_receptus_spelling(client):
    nfc = lambda x: unicodedata.normalize("NFC", x)       # comparison only; the page itself is not normalized
    html = client.get("/read/Jhn/1").get_data(as_text=True)
    assert html.count('class="verse') == 51 and nfc("λόγος") in nfc(html)
    mat = nfc(client.get("/read/Mat/1").get_data(as_text=True))
    assert nfc("Δαβὶδ") in mat and nfc("Δαυὶδ") not in mat       # Scrivener spelling, not the critical text


def test_word_api_hebrew(client):
    d = client.get("/api/word/heb/Gen.1.1/1").get_json()
    assert d["strong"] == "H7225" and d["lexicon"]["word"] and d["occurrences"] > 40
    assert len(d["grammar_parts"]) == 2 and all(p["text"] for p in d["grammar_parts"])


def test_word_api_greek(client):
    d = client.get("/api/word/grc/Jhn.1.1/5").get_json()
    assert d["strong"] == "G3056" and d["occurrences"] == 330
    assert "Noun" in d["grammar_parts"][0]["text"] and d["lexicon"]["meaning"]
    assert d["translit"] == "logos"


def test_verse_vocabulary(client):
    d = client.get("/api/verse/grc/Jhn.1.1").get_json()
    assert "G3056" in [w["strong"] for w in d["words"]]


def test_strongs_search_and_paging(client):
    r = client.get("/search?q=G3056")
    html = r.get_data(as_text=True)
    assert r.status_code == 200 and "330 occurrences" in html and 'class="w plain match"' in html
    assert "page 1 of" in html
    assert client.get("/search?q=g%203056").status_code == 200       # forgiving input
    assert "H7225" in client.get("/search?q=H7225").get_data(as_text=True)


def test_text_search_without_english_explains_itself(client):
    assert "needs an English text" in client.get("/search?q=love").get_data(as_text=True)


def test_bad_input_is_handled(client):
    assert client.get("/read/Gen/999").status_code == 404
    assert client.get("/read/Xyz/1").status_code == 404
    assert client.get("/api/word/heb/Gen.1.1/9999").status_code == 404
    html = client.get("/search?q=%3Cscript%3Ealert(1)%3C/script%3E").get_data(as_text=True)
    assert "<script>alert(1)" not in html


def test_lexicon_html_is_sanitized():
    out = appmod.clean_html("<script>x()</script><b onclick='x()'>bold</b><img src=x onerror=y>"
                            "<ref='Psa.2.1-12'>Psa.2:1</ref>")
    assert "<script" not in out and "onclick" not in out and "<img" not in out
    assert '<a class="xref" href="/read/Psa/2#v1">Psa.2:1</a>' in out and "<b>bold</b>" in out


def test_greek_transliteration():
    t = appmod.translit_greek
    assert t("λόγος") == "logos" and t("Ἰησοῦς") == "Iēsous" and t("ἁγίος") == "hagios"
    assert t("ῥῆμα") == "rhēma" and t("Ἀβραάμ") == "Abraam" and t("ἄγγελος") == "angelos"
    assert t("αἷμα") == "haima"


def test_english_import_and_display(tmp_path, monkeypatch, client):
    """Uses a clearly synthetic fixture: the test text is not scripture."""
    copy = tmp_path / "t.db"
    shutil.copy(appmod.DB_PATH, copy)
    tsv = tmp_path / "TEST.tsv"
    tsv.write_text("Gen.1.1\tsynthetic test verse about beginnings\nGen.1.2\tsecond synthetic verse\n", encoding="utf-8")
    subprocess.run([sys.executable, os.path.join(HERE, "import_english.py"), "--db", str(copy), "--version", "TEST",
                    "--name", "Test", "--license", "none", "--tsv", str(tsv)], check=True, cwd=HERE,
                   capture_output=True)
    monkeypatch.setattr(appmod, "DB_PATH", str(copy))
    c = appmod.app.test_client()
    html = c.get("/read/Gen/1?v=TEST").get_data(as_text=True)
    assert "synthetic test verse about beginnings" in html
    assert "synthetic test verse" not in c.get("/read/Gen/1").get_data(as_text=True)       # off unless selected
    assert "beginnings" in c.get("/search?q=beginning").get_data(as_text=True)              # stemmed full-text hit
    assert "Gen.1.1" not in c.get("/read/Gen/1?v=NOPE").get_data(as_text=True).split("verse", 1)[0]
