const promptInput = document.getElementById("promptInput");
const runButton = document.getElementById("runButton");
const newChatButton = document.getElementById("newChatButton");
const toggleTraceButton = document.getElementById("toggleTraceButton");
const closeTraceButton = document.getElementById("closeTraceButton");

const statusTitle = document.getElementById("statusTitle");
const statusPill = document.getElementById("statusPill");
const latencyLabel = document.getElementById("latencyLabel");
const rawTrace = document.getElementById("rawTrace");

const plannerOutput = document.getElementById("plannerOutput");
const researcherOutput = document.getElementById("researcherOutput");
const generatorOutput = document.getElementById("generatorOutput");
const verifierOutput = document.getElementById("verifierOutput");

const messageList = document.getElementById("messageList");
const chatEmptyState = document.getElementById("chatEmptyState");
const suggestionsPanel = document.getElementById("suggestionsPanel");
const tracePanel = document.getElementById("tracePanel");
const copyTraceButton = document.getElementById("copyTraceButton");
const chatStage = document.querySelector(".chat-stage");
const addModelButton = document.getElementById("addModelButton");
const composerAddModelButton = document.getElementById("composerAddModelButton");
const modelDialog = document.getElementById("modelDialog");
const closeModelDialogButton = document.getElementById("closeModelDialogButton");
const modelPluginForm = document.getElementById("modelPluginForm");
const modelPluginStatus = document.getElementById("modelPluginStatus");
const modelPill = document.getElementById("modelPill");

const messageHistory = [];

function formatJson(value) {
  if (value === null || value === undefined) {
    return "null";
  }
  return JSON.stringify(value, null, 2);
}

async function copyText(value) {
  await navigator.clipboard.writeText(value);
}

function formatPipelineValue(value) {
  if (value === null || value === undefined) {
    return "Skipped.";
  }
  if (typeof value === "string") {
    return value || "No output.";
  }
  return JSON.stringify(value, null, 2);
}

function buildPipelineSteps(trace) {
  if (!trace) {
    return [];
  }
  return [
    ["Planner Response", trace.planner],
    ["Researcher Response", trace.researcher || "Skipped."],
    ["Generator Response", trace.generator],
    ["Verifier Response", trace.verifier],
    ["Final Generated Response", trace.final_response || "No final response returned."],
  ];
}

function appendPipelineTrace(item, trace) {
  const steps = buildPipelineSteps(trace);
  if (steps.length === 0) {
    return;
  }

  const panel = document.createElement("section");
  panel.className = "message-pipeline";

  for (const [title, value] of steps) {
    const step = document.createElement("details");
    step.className = "message-pipeline-step";
    step.open = title === "Final Generated Response";

    const summary = document.createElement("summary");
    summary.textContent = title;

    const output = document.createElement("pre");
    output.textContent = formatPipelineValue(value);

    step.append(summary, output);
    panel.appendChild(step);
  }

  item.appendChild(panel);
}

function renderMessages() {
  messageList.innerHTML = "";
  chatStage.classList.toggle("has-messages", messageHistory.length > 0);

  if (messageHistory.length === 0) {
    messageList.hidden = true;
    chatEmptyState.hidden = false;
    suggestionsPanel.hidden = false;
    return;
  }

  chatEmptyState.hidden = true;
  suggestionsPanel.hidden = true;
  messageList.hidden = false;

  for (const message of messageHistory) {
    const item = document.createElement("article");
    item.className = `message-bubble ${message.role}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = message.role === "user" ? "You" : message.label || "Guardrail";

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.content;

    item.append(meta, content);

    if (message.role === "assistant" && message.trace) {
      appendPipelineTrace(item, message.trace);
    }

    if (message.footnote) {
      const footnote = document.createElement("div");
      footnote.className = "message-footnote";
      footnote.textContent = message.footnote;
      item.appendChild(footnote);
    }

    if (message.role === "assistant" && message.generationId) {
      const feedback = document.createElement("div");
      feedback.className = "message-feedback";

      const approveButton = document.createElement("button");
      approveButton.type = "button";
      approveButton.textContent = "Good for DPO";
      approveButton.disabled = Boolean(message.feedbackStatus);
      approveButton.addEventListener("click", () => sendDpoPreference(message.generationId, "chosen"));

      const rejectButton = document.createElement("button");
      rejectButton.type = "button";
      rejectButton.textContent = "Bad for DPO";
      rejectButton.disabled = Boolean(message.feedbackStatus);
      rejectButton.addEventListener("click", () => sendDpoPreference(message.generationId, "rejected"));

      feedback.append(approveButton, rejectButton);
      if (message.feedbackStatus) {
        const status = document.createElement("span");
        status.textContent = message.feedbackStatus;
        feedback.appendChild(status);
      }
      item.appendChild(feedback);
    }

    messageList.appendChild(item);
  }

  messageList.scrollTop = messageList.scrollHeight;
}

function setBusy(isBusy) {
  runButton.disabled = isBusy;
  runButton.textContent = isBusy ? "Thinking..." : "Send";
}

function openModelDialog() {
  modelDialog.showModal();
}

async function loadGeneratorProvider() {
  try {
    const response = await fetch("/generator/provider");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Unable to load generator provider");
    }
    const runtime = data.runtime || {};
    const label = runtime.resolved_model_name || runtime.model_name || data.settings?.model_name || data.active_provider;
    modelPill.textContent = label ? `Generator: ${label}` : "Guardrailed Model";
  } catch {
    modelPill.textContent = "Generator: unavailable";
  }
}

async function saveGeneratorProvider(event) {
  event.preventDefault();
  const formData = new FormData(modelPluginForm);
  const providerType = formData.get("type");
  const payload = {
    type: providerType,
    base_url: formData.get("base_url") || undefined,
    model_name: formData.get("model_name") || undefined,
    api_key: formData.get("api_key") || undefined,
    auto_discover_model: !formData.get("model_name"),
    request_timeout: 600,
  };

  if (providerType === "ollama" && !payload.base_url) {
    payload.base_url = "http://127.0.0.1:11434";
  }
  if ((providerType === "lm_studio" || providerType === "openai_compatible") && !payload.base_url) {
    payload.base_url = "http://127.0.0.1:1234/v1";
  }
  if (providerType === "custom_http") {
    payload.endpoint = payload.base_url;
  }
  if (providerType === "lm_studio" && !payload.api_key) {
    payload.api_key = "lm-studio";
  }

  modelPluginStatus.textContent = "Activating generator plugin...";
  try {
    const response = await fetch("/generator/provider", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Unable to activate provider");
    }
    const runtime = data.runtime || {};
    const label = runtime.resolved_model_name || runtime.model_name || payload.model_name || data.active_provider;
    modelPill.textContent = label ? `Generator: ${label}` : "Guardrailed Model";
    modelPluginStatus.textContent = "Generator model plugin activated.";
    modelDialog.close();
  } catch (error) {
    modelPluginStatus.textContent = error.message;
  }
}

function setTraceOpen(isOpen) {
  tracePanel.classList.toggle("open", isOpen);
}

function pushUserMessage(prompt) {
  messageHistory.push({
    role: "user",
    content: prompt,
  });
  renderMessages();
}

function findMessageByGenerationId(generationId) {
  return messageHistory.find((message) => message.generationId === generationId);
}

async function sendDpoPreference(generationId, preference) {
  const message = findMessageByGenerationId(generationId);
  if (message) {
    message.feedbackStatus = "Saving preference...";
    renderMessages();
  }

  try {
    const response = await fetch("/dpo/preference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ generation_id: generationId, preference }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Unable to save DPO preference");
    }
    if (message) {
      message.feedbackStatus = "Preference saved";
    }
  } catch (error) {
    if (message) {
      message.feedbackStatus = error.message;
    }
  }
  renderMessages();
}

function upsertAssistantMessage(content, footnote = "", dpo = null, trace = null) {
  const previous = messageHistory[messageHistory.length - 1];
  if (previous && previous.role === "assistant" && previous.pending) {
    previous.content = content;
    previous.pending = false;
    previous.footnote = footnote;
    previous.generationId = dpo?.generationId || "";
    previous.dpoPrompt = dpo?.prompt || "";
    previous.feedbackStatus = "";
    previous.trace = trace;
  } else {
    messageHistory.push({
      role: "assistant",
      label: "Guardrailed Response",
      content,
      footnote,
      generationId: dpo?.generationId || "",
      dpoPrompt: dpo?.prompt || "",
      feedbackStatus: "",
      trace,
      pending: false,
    });
  }
  renderMessages();
}

function addPendingAssistantMessage() {
  messageHistory.push({
    role: "assistant",
    label: "Guardrailed Response",
    content: "Working through planner, researcher, generator, and verifier...",
    footnote: "Trace will update as soon as the pipeline returns.",
    pending: true,
  });
  renderMessages();
}

function resetTracePanels() {
  plannerOutput.textContent = "Waiting for input...";
  researcherOutput.textContent = "Waiting for input...";
  generatorOutput.textContent = "Waiting for input...";
  verifierOutput.textContent = "Waiting for input...";
  rawTrace.textContent = "{}";
}

async function runPipeline(prefilledPrompt) {
  const prompt = (prefilledPrompt ?? promptInput.value).trim();
  if (!prompt) {
    statusTitle.textContent = "Prompt Required";
    statusPill.textContent = "No input";
    return;
  }

  if (!prefilledPrompt) {
    promptInput.value = "";
  }

  pushUserMessage(prompt);
  addPendingAssistantMessage();
  setBusy(true);
  setTraceOpen(true);
  statusTitle.textContent = "Running guardrail pipeline";
  statusPill.textContent = "Working";
  latencyLabel.textContent = "Request in progress...";

  const startedAt = performance.now();

  try {
    const response = await fetch("/guardrail_trace", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ prompt }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Request failed with status ${response.status}`);
    }

    const elapsedMs = Math.round(performance.now() - startedAt);
    const finalStatus = data.final_status || "Completed";
    const finalResponse = data.final_response || "";

    statusTitle.textContent = finalStatus;
    statusPill.textContent = finalStatus;
    latencyLabel.textContent = `Completed in ${elapsedMs} ms`;

    const footnote = `Planner ${data.planner?.decision || "n/a"} | Verifier ${data.verifier?.verdict || "n/a"}`;
    upsertAssistantMessage(finalResponse || "No response returned.", footnote, {
      generationId: data.dpo_generation?.id || "",
      prompt: data.dpo_generation?.prompt || prompt,
    }, data);

    rawTrace.textContent = formatJson(data);
    plannerOutput.textContent = formatJson(data.planner);
    researcherOutput.textContent = data.researcher ? formatJson(data.researcher) : "Skipped.";
    generatorOutput.textContent = formatJson(data.generator);
    verifierOutput.textContent = formatJson(data.verifier);
  } catch (error) {
    statusTitle.textContent = "Request Failed";
    statusPill.textContent = "Error";
    latencyLabel.textContent = "Check whether all services are running.";
    upsertAssistantMessage(error.message, "The pipeline did not complete successfully.");
    rawTrace.textContent = formatJson({ error: error.message });
    plannerOutput.textContent = "Unavailable";
    researcherOutput.textContent = "Unavailable";
    generatorOutput.textContent = "Unavailable";
    verifierOutput.textContent = "Unavailable";
  } finally {
    setBusy(false);
  }
}

document.querySelectorAll(".chat-suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    promptInput.value = button.dataset.prompt || "";
    runPipeline(button.dataset.prompt || "");
  });
});

runButton.addEventListener("click", () => runPipeline());
addModelButton.addEventListener("click", openModelDialog);
composerAddModelButton.addEventListener("click", openModelDialog);
closeModelDialogButton.addEventListener("click", () => modelDialog.close());
modelPluginForm.addEventListener("submit", saveGeneratorProvider);
newChatButton.addEventListener("click", () => {
  messageHistory.length = 0;
  promptInput.value = "";
  statusTitle.textContent = "Ready to test your LLM";
  statusPill.textContent = "Idle";
  latencyLabel.textContent = "No request yet";
  resetTracePanels();
  renderMessages();
});
toggleTraceButton.addEventListener("click", () => {
  setTraceOpen(!tracePanel.classList.contains("open"));
});
closeTraceButton.addEventListener("click", () => setTraceOpen(false));
copyTraceButton.addEventListener("click", () => copyText(rawTrace.textContent));

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    runPipeline();
  }
});

resetTracePanels();
renderMessages();
loadGeneratorProvider();
