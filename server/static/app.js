const fallbackSampleQuestions = [
  "How much Burger Bun was used in Laxmi Nagar Store yesterday?",
  "Which 5 stores had the highest D-1 quantity for Paper Napkin on 2026-04-04?",
  "Show city-wise D-1 quantity for Burger Bun on 2026-04-04.",
  "Which items have the highest variance in Laxmi Nagar Store?",
  "Has variance increased compared to yesterday for Burger Bun in Laxmi Nagar Store?",
  "What is the current stock level of Burger Bun in Laxmi Nagar Store?",
];

const elements = {
  datasetPath: document.getElementById("datasetPath"),
  availableDates: document.getElementById("availableDates"),
  supportedList: document.getElementById("supportedList"),
  unsupportedList: document.getElementById("unsupportedList"),
  sampleList: document.getElementById("sampleList"),
  queryForm: document.getElementById("queryForm"),
  questionInput: document.getElementById("questionInput"),
  maxRowsInput: document.getElementById("maxRowsInput"),
  askButton: document.getElementById("askButton"),
  answerPanel: document.getElementById("answerPanel"),
  statusChip: document.getElementById("statusChip"),
  answerText: document.getElementById("answerText"),
  parsedIntent: document.getElementById("parsedIntent"),
  appliedFilters: document.getElementById("appliedFilters"),
  notesBox: document.getElementById("notesBox"),
  notesList: document.getElementById("notesList"),
  answerTableWrap: document.getElementById("answerTableWrap"),
  evidenceTableWrap: document.getElementById("evidenceTableWrap"),
};

function setBusy(isBusy) {
  elements.askButton.disabled = isBusy;
  elements.askButton.textContent = isBusy ? "Loading..." : "Get Answer";
}

function setStatus(text, kind = "") {
  elements.statusChip.textContent = text;
  elements.statusChip.className = "status-chip";
  if (kind) {
    elements.statusChip.classList.add(kind);
  }
}

function renderList(container, values) {
  container.innerHTML = "";
  values.forEach((value) => {
    const item = document.createElement("li");
    item.textContent = value;
    container.appendChild(item);
  });
}

function renderSamples(questions) {
  elements.sampleList.innerHTML = "";
  questions.forEach((question) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-button";
    button.textContent = question;
    button.addEventListener("click", () => {
      elements.questionInput.value = question;
      elements.questionInput.focus();
    });
    elements.sampleList.appendChild(button);
  });
  if (!elements.questionInput.value.trim() && questions.length > 0) {
    elements.questionInput.value = questions[0];
  }
}

function renderTable(container, rows) {
  if (!rows || rows.length === 0) {
    container.innerHTML = '<p class="empty-state">No rows returned.</p>';
    return;
  }

  const columns = Object.keys(rows[0]);
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");

  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    headRow.appendChild(th);
  });

  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      const value = row[column];
      td.textContent = value === null || value === undefined ? "-" : String(value);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  container.innerHTML = "";
  container.appendChild(table);
}

function renderNotes(notes) {
  if (!notes || notes.length === 0) {
    elements.notesBox.hidden = true;
    elements.notesList.innerHTML = "";
    return;
  }

  elements.notesBox.hidden = false;
  elements.notesList.innerHTML = "";
  notes.forEach((note) => {
    const item = document.createElement("li");
    item.textContent = note;
    elements.notesList.appendChild(item);
  });
}

function formatFilters(filters) {
  const entries = Object.entries(filters || {});
  if (entries.length === 0) {
    return "-";
  }
  return entries.map(([key, value]) => `${key}=${value}`).join(", ");
}

async function loadCapabilities() {
  const response = await fetch("/office/capabilities");
  const payload = await response.json();
  elements.datasetPath.textContent = payload.dataset_path;
  elements.availableDates.textContent = payload.available_dates.join(", ");
  renderSamples(payload.sample_questions && payload.sample_questions.length ? payload.sample_questions : fallbackSampleQuestions);
  renderList(elements.supportedList, payload.supported_patterns);
  renderList(elements.unsupportedList, payload.unsupported_examples);
}

async function submitQuestion(event) {
  event.preventDefault();
  const question = elements.questionInput.value.trim();
  if (!question) {
    setStatus("Please enter a question first.", "unsupported");
    return;
  }

  setBusy(true);
  setStatus("Looking up the answer...", "loading");
  elements.answerPanel.scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const response = await fetch("/office/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        max_rows: Number(elements.maxRowsInput.value || 10),
      }),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.detail ? JSON.stringify(errorPayload.detail) : `HTTP ${response.status}`);
    }

    const payload = await response.json();
    elements.answerText.textContent = payload.answer;
    elements.parsedIntent.textContent = payload.parsed_intent || "-";
    elements.appliedFilters.textContent = formatFilters(payload.applied_filters);
    renderNotes(payload.notes);
    renderTable(elements.answerTableWrap, payload.table);
    renderTable(elements.evidenceTableWrap, payload.evidence_rows);
    setStatus(
      payload.supported ? "Answer ready" : "Question outside current dataset scope",
      payload.supported ? "supported" : "unsupported",
    );
  } catch (error) {
    elements.answerText.textContent = `The assistant could not reach the backend cleanly. ${error.message || ""}`.trim();
    elements.parsedIntent.textContent = "-";
    elements.appliedFilters.textContent = "-";
    renderNotes([]);
    renderTable(elements.answerTableWrap, []);
    renderTable(elements.evidenceTableWrap, []);
    setStatus("Request failed", "unsupported");
    console.error(error);
  } finally {
    setBusy(false);
  }
}

async function init() {
  elements.queryForm.addEventListener("submit", submitQuestion);
  try {
    await loadCapabilities();
    setStatus("Ready");
  } catch (error) {
    renderSamples(fallbackSampleQuestions);
    setStatus("Setup information failed to load", "unsupported");
    console.error(error);
  }
}

init();
