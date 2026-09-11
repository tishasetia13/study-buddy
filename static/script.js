// ---- Session ID ----
// Every visitor gets their own random ID so their uploaded PDF (and the
// embeddings generated from it) never gets mixed up with anyone else's.
// We generate it once per browser with crypto.randomUUID() and save it in
// localStorage so it survives page reloads/tab closes.
let sessionId = localStorage.getItem("studyBuddySessionId");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("studyBuddySessionId", sessionId);
}

const chatLog = document.getElementById("chat-log");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const pdfInput = document.getElementById("pdf-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadStatus = document.getElementById("upload-status");

// The chat box stays disabled until a PDF has been successfully processed
// in THIS page load. We don't trust localStorage alone to mean "ready" -
// the server keeps session data on disk/in memory, and none of that
// survives a server restart, so we only trust an upload we just confirmed.
let documentReady = false;

async function uploadPdf() {
  const file = pdfInput.files[0];
  if (!file) {
    uploadStatus.textContent = "Please choose a PDF file first.";
    uploadStatus.className = "status-error";
    return;
  }

  uploadBtn.disabled = true;
  uploadStatus.textContent = "Parsing your document... (the first upload can take a bit longer while the AI model loads)";
  uploadStatus.className = "status-pending";

  // multipart/form-data, not JSON - that's what lets us attach the raw PDF bytes.
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server returned ${response.status}`);
    }

    const data = await response.json();

    // On a brand-new browser, the server may have generated this session_id
    // for us (rather than us sending a real one) - save whatever it returns
    // so every future /ask call uses the same session.
    sessionId = data.session_id;
    localStorage.setItem("studyBuddySessionId", sessionId);

    uploadStatus.textContent = `Ready! Loaded "${data.filename}" (${data.num_chunks} chunks). Ask away below.`;
    uploadStatus.className = "status-ready";

    documentReady = true;
    questionInput.disabled = false;
    questionInput.placeholder = "Ask something about your notes...";
    sendBtn.disabled = false;

  } catch (err) {
    uploadStatus.textContent = `Upload failed: ${err.message}`;
    uploadStatus.className = "status-error";
    console.error(err);
  } finally {
    uploadBtn.disabled = false;
  }
}

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  if (!documentReady) {
    appendMessage("Please upload a PDF before asking a question.", "answer-msg");
    return;
  }

  // Show the user's question immediately
  appendMessage(question, "user-msg");
  questionInput.value = "";
  sendBtn.disabled = true;
  sendBtn.textContent = "Thinking...";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question: question }),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Server returned ${response.status}`);
    }

    const data = await response.json();
    renderAnswer(data);

  } catch (err) {
    appendMessage(`Error: ${err.message}`, "answer-msg");
    console.error(err);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Send";
  }
}

function appendMessage(text, className) {
  const div = document.createElement("div");
  div.className = `message ${className}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderAnswer(data) {
  const div = document.createElement("div");
  div.className = "message answer-msg";

  const answerP = document.createElement("p");
  answerP.textContent = data.answer;
  div.appendChild(answerP);

  if (data.sources && data.sources.length > 0) {
    const citeDiv = document.createElement("div");
    citeDiv.className = "citations";
    citeDiv.textContent = "Sources: " + data.sources
      .map(s => `${s.source} (p.${s.page})`)
      .join(", ");
    div.appendChild(citeDiv);
  }

  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

uploadBtn.addEventListener("click", uploadPdf);
sendBtn.addEventListener("click", sendQuestion);
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendQuestion();
});
