/* Elder Scam Checker -- progressive enhancement.
 * The form works fine as a plain POST with no JS (see app.py /check, which
 * renders _result.html server-side). This script upgrades it to submit via
 * fetch (no page reload) and adds a few senior-friendly extras: adjustable
 * text size, example messages, and a "read result aloud" button.
 */
(function () {
  "use strict";

  var form = document.getElementById("checker-form");
  var textarea = document.getElementById("message");
  var senderInput = document.getElementById("sender");
  var resultContainer = document.getElementById("result-container");
  var clearBtn = document.getElementById("clear-btn");
  var checkBtn = document.getElementById("check-btn");

  // Same shapes as templates/_icons.html -- kept in sync by hand since this
  // path renders results without a page reload (fetch), so it can't just
  // include the Jinja macro. Shape-coded (circle/triangle/octagon), not just
  // color-coded, matching universal road-sign conventions.
  var ICON_SVG = {
    low: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9.25"/><path d="M7.8 12.4 L10.4 15 L16.2 8.6"/></svg>',
    medium: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.7 L22.3 20.7 H1.7 Z"/><line x1="12" y1="9.3" x2="12" y2="14.3"/><circle cx="12" cy="17.4" r="1.05" fill="currentColor" stroke="none"/></svg>',
    high: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7.8 2.5 H16.2 L21.5 7.8 V16.2 L16.2 21.5 H7.8 L2.5 16.2 V7.8 Z"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>'
  };
  var SPEAKER_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9.5 H8 L13 5 V19 L8 14.5 H4 Z"/><path d="M16.5 8.5 C18 10 18 14 16.5 15.5"/><path d="M19 6 C21.5 8.7 21.5 15.3 19 18"/></svg>';

  var EXAMPLES = {
    scam: "Grandma, it's me, please don't hang up. I was in a car accident and I'm at the police station. I need $1,500 for bail today but please don't tell Mom and Dad, I'm so embarrassed. Can you send it as gift cards?",
    safe: "Hi Dad, just checking in on your doctor's appointment today. Call me when you get a chance, love you!"
  };

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function renderResult(result) {
    var icon = ICON_SVG[result.risk_level] || "";

    var html = '<div class="result-card result-' + result.risk_level + '">';
    html += '<div class="result-headline">';
    html += '<span class="result-icon">' + icon + '</span>';
    html += '<h2 class="result-verdict">' + escapeHtml(result.risk_label) + '</h2>';
    html += '</div>';

    if (result.likely_type) {
      html += '<p class="result-subtitle">This matches the pattern of a <strong>' +
        escapeHtml(result.likely_type.label) + '</strong>.</p>';
    }

    var senderSignals = result.sender_signals || [];
    var suspiciousSender = senderSignals.filter(function (s) { return s.type === "suspicious"; });
    var reassuringSender = senderSignals.filter(function (s) { return s.type === "reassuring"; });
    var infoSender = senderSignals.filter(function (s) { return s.type === "info"; });

    html += '<div class="result-section"><h3>Why we flagged this</h3>';
    if ((result.tactics && result.tactics.length) || suspiciousSender.length) {
      html += '<ul class="flag-list">';
      (result.tactics || []).forEach(function (t) {
        html += '<li>' + escapeHtml(t.label) + '</li>';
      });
      suspiciousSender.forEach(function (s) {
        html += '<li>' + escapeHtml(s.label) + '</li>';
      });
      html += '</ul>';
    } else if (reassuringSender.length) {
      html += '<p>We didn’t find common scam warning signs in the wording, and the sender looks legitimate:</p>';
      html += '<ul class="flag-list flag-list-good">';
      reassuringSender.forEach(function (s) {
        html += '<li>' + escapeHtml(s.label) + '</li>';
      });
      html += '</ul>';
    } else {
      html += '<p>We didn’t find common scam warning signs in the wording. Still, if you weren’t expecting this message, it’s okay to double check.</p>';
    }
    html += '</div>';

    if (infoSender.length) {
      html += '<div class="result-section"><p class="sender-note">' + escapeHtml(infoSender[0].label) + '</p></div>';
    }

    if (result.likely_type) {
      html += '<div class="result-section advice-box"><h3>What to do</h3><p>' +
        escapeHtml(result.likely_type.advice) + '</p></div>';
    }

    html += '<button type="button" class="btn btn-secondary" id="check-another">Check Another Message</button>';

    if ("speechSynthesis" in window) {
      html += ' <button type="button" class="btn btn-plain btn-with-icon" id="read-aloud" style="margin-top:1.4rem;">' +
        '<span class="btn-icon">' + SPEAKER_SVG + '</span> Read This Aloud</button>';
    }

    html += '</div>';

    resultContainer.innerHTML = html;
    resultContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    wireResultButtons(result);
  }

  function speak(result) {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    var parts = [result.risk_label + "."];
    if (result.likely_type) {
      parts.push("This matches the pattern of a " + result.likely_type.label + ".");
    }
    var senderSignals = result.sender_signals || [];
    var flagLabels = (result.tactics || []).map(function (t) { return t.label; })
      .concat(senderSignals.filter(function (s) { return s.type === "suspicious"; }).map(function (s) { return s.label; }));
    if (flagLabels.length) {
      parts.push("Why we flagged this: " + flagLabels.join(". ") + ".");
    }
    if (result.likely_type) {
      parts.push("What to do: " + result.likely_type.advice);
    }
    var utterance = new SpeechSynthesisUtterance(parts.join(" "));
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  }

  function wireResultButtons(result) {
    var again = document.getElementById("check-another");
    if (again) {
      again.addEventListener("click", function () {
        resultContainer.innerHTML = "";
        textarea.value = "";
        if (senderInput) senderInput.value = "";
        textarea.focus();
      });
    }
    var readAloud = document.getElementById("read-aloud");
    if (readAloud && result) {
      readAloud.addEventListener("click", function () { speak(result); });
    }
  }

  // Enhance the initial server-rendered result (no-JS fallback path), if present.
  var existingCard = resultContainer.querySelector(".result-card");
  if (existingCard) {
    wireResultButtons(null);
  }

  if (form) {
    form.addEventListener("submit", function (evt) {
      var text = textarea.value.trim();
      if (!text) return; // let normal HTML validation / server handle it

      evt.preventDefault();
      checkBtn.disabled = true;
      checkBtn.textContent = "Checking…";

      fetch(form.action, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Requested-With": "fetch"
        },
        body: JSON.stringify({ message: text, sender: senderInput ? senderInput.value.trim() : "" })
      })
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
          if (data.error) return;
          renderResult(data);
        })
        .catch(function () {
          // Network hiccup: fall back to a normal form submit.
          form.submit();
        })
        .finally(function () {
          checkBtn.disabled = false;
          checkBtn.textContent = "Check This Message";
        });
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", function () {
      textarea.value = "";
      if (senderInput) senderInput.value = "";
      resultContainer.innerHTML = "";
      textarea.focus();
    });
  }

  document.querySelectorAll(".btn-example").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var kind = btn.getAttribute("data-example");
      textarea.value = EXAMPLES[kind] || "";
      textarea.focus();
    });
  });

  // ---- Adjustable text size (remembered between visits) ----
  var root = document.documentElement;
  var STORAGE_KEY = "scamCheckerFontScale";
  var scale = parseFloat(localStorage.getItem(STORAGE_KEY)) || 1;

  function applyScale() {
    root.style.setProperty("--font-scale", scale);
    localStorage.setItem(STORAGE_KEY, scale);
  }
  applyScale();

  var upBtn = document.getElementById("text-size-up");
  var downBtn = document.getElementById("text-size-down");
  if (upBtn) {
    upBtn.addEventListener("click", function () {
      scale = Math.min(scale + 0.1, 1.6);
      applyScale();
    });
  }
  if (downBtn) {
    downBtn.addEventListener("click", function () {
      scale = Math.max(scale - 0.1, 0.85);
      applyScale();
    });
  }
})();
