/* ElRoi-Bible front end. Plain JS, no dependencies. */
(function () {
  var root = document.documentElement;
  var panel = document.getElementById("panel");
  var body = document.getElementById("panel-body");

  /* theme and scripture size: same localStorage keys as El Roi */
  var theme = document.getElementById("theme"), size = document.getElementById("size");
  if (theme) {
    theme.value = root.getAttribute("data-theme") || "light";
    theme.addEventListener("change", function () {
      root.setAttribute("data-theme", theme.value);
      try { localStorage.setItem("theme", theme.value); } catch (e) {}
    });
  }
  if (size) {
    size.value = root.getAttribute("data-scripture-size") || "medium";
    size.addEventListener("change", function () {
      root.setAttribute("data-scripture-size", size.value);
      try { localStorage.setItem("scripture_size", size.value); } catch (e) {}
    });
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function section(title) { body.appendChild(el("h3", null, title)); }
  function open() { panel.hidden = false; }
  function close() {
    panel.hidden = true;
    document.querySelectorAll(".w.sel").forEach(function (n) { n.classList.remove("sel"); });
  }
  panel.querySelector(".close").addEventListener("click", close);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") close(); });

  function getJSON(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    });
  }

  function showWord(node) {
    document.querySelectorAll(".w.sel").forEach(function (n) { n.classList.remove("sel"); });
    node.classList.add("sel");
    var lang = node.dataset.lang;
    getJSON("/api/word/" + lang + "/" + node.dataset.ref + "/" + node.dataset.pos).then(function (d) {
      body.textContent = "";
      body.appendChild(el("div", "big " + lang, d.surface));
      var meta = [d.translit, d.gloss].filter(Boolean).join(" · ");
      if (meta) body.appendChild(el("div", null, meta));
      if (d.strong) {
        var a = el("a", "pill", d.strong);
        a.href = "/search?q=" + d.strong;
        var p = el("p");
        p.appendChild(a);
        p.appendChild(document.createTextNode("  " + d.occurrences + " occurrences"));
        body.appendChild(p);
      }
      if (d.grammar_parts && d.grammar_parts.length) {
        section("Parsing");
        d.grammar_parts.forEach(function (g) {
          var row = el("div");
          row.appendChild(el("span", "pill", g.code));
          row.appendChild(document.createTextNode(" " + (g.text || "")));
          body.appendChild(row);
        });
      }
      if (d.lexicon) {
        section("Lexicon");
        var head = [d.lexicon.word, d.lexicon.translit, d.lexicon.gloss].filter(Boolean).join(" · ");
        body.appendChild(el("div", null, head));
        var m = el("div", "meaning");
        m.innerHTML = d.lexicon.meaning;          /* sanitized on the server */
        body.appendChild(m);
      }
      if (d.glosses && d.glosses.length) {
        section("How STEP glosses this word");
        var ul = el("ul");
        d.glosses.forEach(function (g) { ul.appendChild(el("li", null, g.gloss + " (" + g.count + ")")); });
        body.appendChild(ul);
      }
      open();
    }).catch(function () {
      body.textContent = "Could not load that word.";
      open();
    });
  }

  function showVerse(btn) {
    getJSON("/api/verse/" + btn.dataset.lang + "/" + btn.dataset.ref).then(function (d) {
      body.textContent = "";
      body.appendChild(el("div", "big", d.ref.replace(/\./g, " ").replace(/^(\S+) /, "$1 ")));
      section("Vocabulary of this verse");
      var ul = el("ul");
      d.words.forEach(function (w) {
        var li = el("li");
        var a = el("a", null, w.strong);
        a.href = "/search?q=" + w.strong;
        li.appendChild(a);
        li.appendChild(document.createTextNode("  " + [w.lemma || w.surface, w.gloss].filter(Boolean).join(" · ")));
        ul.appendChild(li);
      });
      body.appendChild(ul);
      open();
    });
  }

  document.addEventListener("click", function (e) {
    var w = e.target.closest(".w"), v = e.target.closest(".vnum");
    if (w) showWord(w);
    else if (v) showVerse(v);
  });
  document.addEventListener("keydown", function (e) {
    if ((e.key === "Enter" || e.key === " ") && e.target.classList && e.target.classList.contains("w")) {
      e.preventDefault();
      showWord(e.target);
    }
  });

  /* reader controls: fold the checkboxes into short query parameters and reload */
  var form = document.getElementById("controls");
  if (form) {
    form.addEventListener("change", function () { form.requestSubmit(); });
    form.addEventListener("submit", function () {
      var lines = "", vers = [];
      form.querySelectorAll('input[name="l"]:checked').forEach(function (c) { lines += c.value; });
      form.querySelectorAll('input[name="vv"]:checked').forEach(function (c) { vers.push(c.value); });
      document.getElementById("lines-field").value = lines;
      document.getElementById("v-field").value = vers.join(",");
      form.querySelectorAll('input[name="l"],input[name="vv"]').forEach(function (c) { c.disabled = true; });
    });
  }
})();
