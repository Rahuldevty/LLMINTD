const deployPromptInput = document.getElementById("deployPromptInput");
const deploySendButton = document.getElementById("deploySendButton");
const newDeployChatButton = document.getElementById("newDeployChatButton");
const deployMessageList = document.getElementById("deployMessageList");
const deployEmptyState = document.getElementById("deployEmptyState");
const deploySuggestions = document.getElementById("deploySuggestions");
const deployLatencyLabel = document.getElementById("deployLatencyLabel");
const deployStatusText = document.getElementById("deployStatusText");
const deployStage = document.querySelector(".deploy-stage");
const deployAddModelButton = document.getElementById("deployAddModelButton");
const deployComposerAddModelButton = document.getElementById("deployComposerAddModelButton");
const deployModelDialog = document.getElementById("deployModelDialog");
const deployCloseModelDialogButton = document.getElementById("deployCloseModelDialogButton");
const deployModelPluginForm = document.getElementById("deployModelPluginForm");
const deployModelPluginStatus = document.getElementById("deployModelPluginStatus");
const deployModelPill = document.getElementById("deployModelPill");

const deployMessages = [];

function renderDeployMessages() {
  deployMessageList.innerHTML = "";
  deployStage.classList.toggle("has-messages", deployMessages.length > 0);

  if (deployMessages.length === 0) {
    deployMessageList.hidden = true;
    deployEmptyState.hidden = false;
    deploySuggestions.hidden = false;
    return;
  }

  deployMessageList.hidden = false;
  deployEmptyState.hidden = true;
  deploySuggestions.hidden = true;

  for (const message of deployMessages) {
    const item = document.createElement("article");
    item.className = `message-bubble ${message.role}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = message.role === "user" ? "You" : "Guardrailed LLM";

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.content;

    item.append(meta, content);

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

    deployMessageList.appendChild(item);
  }

  deployMessageList.scrollTop = deployMessageList.scrollHeight;
}

function setDeployBusy(isBusy) {
  deploySendButton.disabled = isBusy;
  deploySendButton.textContent = isBusy ? "Thinking..." : "Send";
}

function openDeployModelDialog() {
  deployModelDialog.showModal();
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
    deployModelPill.textContent = label ? `Generator: ${label}` : "Guardrailed LLM";
  } catch {
    deployModelPill.textContent = "Generator: unavailable";
  }
}

async function saveGeneratorProvider(event) {
  event.preventDefault();
  const formData = new FormData(deployModelPluginForm);
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

  deployModelPluginStatus.textContent = "Activating generator plugin...";
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
    deployModelPill.textContent = label ? `Generator: ${label}` : "Guardrailed LLM";
    deployModelPluginStatus.textContent = "Generator model plugin activated.";
    deployModelDialog.close();
  } catch (error) {
    deployModelPluginStatus.textContent = error.message;
  }
}

function addDeployMessage(role, content) {
  deployMessages.push({ role, content });
  renderDeployMessages();
}

function findDeployMessageByGenerationId(generationId) {
  return deployMessages.find((message) => message.generationId === generationId);
}

async function sendDpoPreference(generationId, preference) {
  const message = findDeployMessageByGenerationId(generationId);
  if (message) {
    message.feedbackStatus = "Saving preference...";
    renderDeployMessages();
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
  renderDeployMessages();
}

function updateLastAssistant(content, generationId = "") {
  const lastMessage = deployMessages[deployMessages.length - 1];
  if (lastMessage && lastMessage.role === "assistant") {
    lastMessage.content = content;
    lastMessage.generationId = generationId;
    lastMessage.feedbackStatus = "";
  } else {
    deployMessages.push({ role: "assistant", content, generationId, feedbackStatus: "" });
  }
  renderDeployMessages();
}

async function sendDeployPrompt(prefilledPrompt) {
  const prompt = (prefilledPrompt ?? deployPromptInput.value).trim();
  if (!prompt) {
    deployStatusText.textContent = "Enter a message first";
    return;
  }

  if (!prefilledPrompt) {
    deployPromptInput.value = "";
  }

  addDeployMessage("user", prompt);
  addDeployMessage("assistant", "Running through the guardrail pipeline...");
  setDeployBusy(true);
  deployStatusText.textContent = "Calling /guardrail";
  deployLatencyLabel.textContent = "Request in progress...";
  const startedAt = performance.now();

  try {
    const response = await fetch("/guardrail", {
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

    updateLastAssistant(data.response || "No response returned.", data.generation_id || "");
    deployStatusText.textContent = "Response passed through guardrail";
    deployLatencyLabel.textContent = `Completed in ${Math.round(performance.now() - startedAt)} ms`;
  } catch (error) {
    updateLastAssistant(error.message);
    deployStatusText.textContent = "Request failed";
    deployLatencyLabel.textContent = "Check services";
  } finally {
    setDeployBusy(false);
  }
}

document.querySelectorAll(".chat-suggestion").forEach((button) => {
  button.addEventListener("click", () => sendDeployPrompt(button.dataset.prompt || ""));
});

document.querySelectorAll("[data-fill]").forEach((button) => {
  button.addEventListener("click", () => {
    deployPromptInput.value = button.dataset.fill || "";
    deployPromptInput.focus();
  });
});

deploySendButton.addEventListener("click", () => sendDeployPrompt());
deployAddModelButton.addEventListener("click", openDeployModelDialog);
deployComposerAddModelButton.addEventListener("click", openDeployModelDialog);
deployCloseModelDialogButton.addEventListener("click", () => deployModelDialog.close());
deployModelPluginForm.addEventListener("submit", saveGeneratorProvider);
newDeployChatButton.addEventListener("click", () => {
  deployMessages.length = 0;
  deployPromptInput.value = "";
  deployStatusText.textContent = "Connected to /guardrail";
  deployLatencyLabel.textContent = "No request yet";
  renderDeployMessages();
});

deployPromptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendDeployPrompt();
  }
});

renderDeployMessages();
loadGeneratorProvider();
